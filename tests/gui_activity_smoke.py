"""Manual end-to-end check of the Transformer Activity panel through the real Qt worker.

FLY_MODEL_DIR=/path/to/small [FLY_TEST_DEVICE=cuda] [FLY_TEST_MONITOR=1] \\
    QT_QPA_PLATFORM=offscreen .venv/bin/python tests/gui_activity_smoke.py
FLY_TEST_MONITOR=1 opens the real audio device at volume 0.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QSettings, Qt, QTimer
from PySide6.QtGui import QFocusEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

out = Path(os.environ.get('FLY_TEST_OUTPUT', '/tmp/fly-trance-activity-test'))
# Keep the user's saved window geometry and panel layout untouched.
QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(out / 'settings'))
from fly_trance.activations import BASE, CX512, MANUAL, RESULT  # noqa: E402
from fly_trance.gui import Observatory  # noqa: E402

app = QApplication([])
w = Observatory(os.environ['FLY_MODEL_DIR'])
w.device.setCurrentText(os.environ.get('FLY_TEST_DEVICE', 'cpu'))
w.monitor.setChecked(os.environ.get('FLY_TEST_MONITOR') == '1')
w.record.setChecked(True)
w.volume.setValue(0)
w.outdir.setText(str(out))
w.show_activity.setChecked(True)
w.activity.more.setChecked(True)
errors, telemetry, snapshots, confirmations = [], [], [], []
stage = {'n': 0, 'hidden_at': None, 'hidden_frames': None, 'shown_at': None}
w.worker.error.connect(errors.append)
w.worker.event.connect(lambda e: confirmations.append(e) if e.get('type', '').startswith(('manual', 'bypass')) else None)
original = w.on_activations
panel, cell = w.activity, {}


def on_activations():
    snapshot = w.worker.activity_slot.take()
    if snapshot is None:
        return
    snapshots.append((time.monotonic(), snapshot))
    w.activity.update_snapshot(snapshot, w.worker.activity_slot.dropped)


w.worker.activations.disconnect(original)
w.worker.activations.connect(on_activations)
# Brain states go through the playback aligner; record when each one is shown.
shown_states = []
_release = w.aligner.release


def release(kind, payload, lag):
    if kind == 'telemetry':
        shown_states.append((payload['sound_sample'], lag, w.worker.generated_samples))
    _release(kind, payload, lag)


w.aligner.release = release


def loaded(_):
    w.fly.setChecked(True)
    w.start_stream()


def target():
    return panel.map._cell_rectf(4, 20).center().toPoint()


def observe(data):
    telemetry.append(data)
    frame, n = data['frame'], stage['n']
    if n == 0 and frame >= 100:
        w.fly.setChecked(False)  # Fly OFF: brain keeps listening, nothing is added
        stage['n'] = 1
    elif n == 1 and frame >= 200:
        w.show_activity.setChecked(False)  # stops collection; stream continues
        stage.update(n=2, hidden_at=len(snapshots), hidden_frames=frame)
    elif n == 2 and frame >= 275:
        w.show_activity.setChecked(True)
        stage.update(n=3, shown_at=len(snapshots))
    elif n == 3 and frame >= 300:
        # Manual Hold with Fly still OFF: manual control works without the fly.
        panel.manual_more.setChecked(True)
        panel.manual_mode.setChecked(True)
        panel.manual_kind.setCurrentIndex(1)
        app.processEvents()
        cell['p'] = target()
        cell['cell'] = panel.map._cell(cell['p'])  # (layer, group) actually under the cursor
        QTest.mousePress(panel.map, Qt.LeftButton, Qt.NoModifier, cell['p'])
        QTest.mouseMove(panel.map, cell['p'] - QPoint(0, 50))  # +0.5 -> +1.0
        stage['n'] = 4
    elif n == 4 and frame >= 330:
        QTest.mouseRelease(panel.map, Qt.LeftButton, Qt.NoModifier, cell['p'] - QPoint(0, 50))
        w.fly.setChecked(True)
        stage['n'] = 5
    elif n == 5 and frame >= 345:
        panel.manual_kind.setCurrentIndex(0)
        QTest.mouseClick(panel.map, Qt.LeftButton, Qt.NoModifier, target())  # one pulse
        panel._bypass_boxes[6].setChecked(True)
        stage['n'] = 6
    elif n == 6 and frame >= 370:
        panel.manual_kind.setCurrentIndex(1)
        QTest.mousePress(panel.map, Qt.LeftButton, Qt.NoModifier, target())
        stage['n'] = 7
    elif n == 7 and frame >= 385:
        # Focus loss while the button is still down must end the hold.
        QApplication.sendEvent(panel.map, QFocusEvent(QEvent.FocusOut, Qt.ActiveWindowFocusReason))
        QTest.mouseRelease(panel.map, Qt.LeftButton, Qt.NoModifier, target())
        stage['n'] = 8
    elif n == 8 and frame >= 400:
        panel.clear_interventions()
        stage['n'] = 9
    elif n == 9 and frame >= 430:
        w.grab().save(str(out / 'activity-window.png'))
        stage['n'] = 10
        w.worker.send('stop')


def recorded(path):
    trace = [json.loads(line) for line in Path(path).with_suffix('.telemetry.jsonl').read_text().splitlines()]
    frames = [r['frame'] for r in trace]
    assert frames == sorted(frames) and len(set(frames)) == len(frames), 'telemetry restarted'
    snaps = [s for _, s in snapshots]
    assert len(snaps) > 10
    assert [s.frame for s in snaps] == sorted(s.frame for s in snaps), 'snapshot frames went backwards'
    for s in snaps:
        assert s.layers == 12 and s.mean.shape == (4, 12, 64) and np.isfinite(s.mean).all() and np.isfinite(s.stats).all()
        if not s.config['enabled']:
            assert np.all(s.stats[CX512] == 0) and np.all(s.mean[CX512] == 0)
    off = [s for s in snaps if not s.config['enabled']]
    on = [s for s in snaps if s.config['enabled']]
    assert off and on and all(s.stats[CX512, s.config['inject'], 0] > 0 for s in on if s.frame > 0)
    hidden = snaps[stage['hidden_at']:stage['shown_at']]
    # At most one snapshot already in flight when the panel was hidden.
    assert len(hidden) <= 1, f'{len(hidden)} snapshots while hidden'
    resumed = snaps[stage['shown_at']:]
    assert resumed and resumed[0].frame > stage['hidden_frames'], 'stream restarted on re-show'
    times = [t for t, s in snapshots[stage['shown_at']:]]
    rate = (len(times) - 1) / (times[-1] - times[0])
    # Manual control, confirmed by the worker with model frames and journaled in the session file.
    kinds = [e['type'] for e in confirmations]
    journal = [e for e in json.loads(Path(path).with_suffix('.json').read_text())['events']
               if e['type'].startswith(('manual', 'bypass'))]
    assert [e['type'] for e in journal] == kinds, (journal, kinds)
    starts = [e for e in journal if e['type'] == 'manual_hold_start']
    ends = [e for e in journal if e['type'] == 'manual_hold_end']
    assert len(starts) == 2 and len(ends) == 2 and [e['reason'] for e in ends] == ['release', 'focus lost']
    assert [e for e in journal if e['type'] == 'manual_pulse'] and journal[-1]['type'] == 'manual_clear'
    assert any(e['type'] == 'bypass' and e['layer'] == 6 and e['enabled'] for e in journal)
    assert journal[-1]['bypass_cleared'] == [6]
    first = starts[0]
    layer, group = cell['cell']
    assert first['layer'] == layer and first['coordinates'] == [16 * group, 16 * group + 16]
    assert first['reference_rms'] > 0 and abs(first['strength'] - 1.0) < 1e-9  # +0.5 dragged 50 px up
    held = [s for s in snaps if any(op[4] == 'hold' for op in s.manual_ops)]
    assert held and all(first['effective_frame'] <= s.frame < ends[0]['effective_frame'] or
                        starts[1]['effective_frame'] <= s.frame < ends[1]['effective_frame'] for s in held)
    assert all(np.flatnonzero(s.rms[MANUAL, layer]).tolist() == [group] for s in held)
    assert not any(s.manual_ops for s in snaps if s.frame >= journal[-1]['effective_frame'])
    bypassed = [s for s in snaps if s.bypass == (6,)]
    assert bypassed and all(np.array_equal(s.mean[BASE, 6], s.mean[RESULT, 5]) for s in bypassed)
    assert '✓' in panel.manual_confirm.text() and 'frame' in panel.manual_confirm.text()
    if w.monitor.isChecked():
        # Sync with audio: shown when its own sound starts, not when it was generated.
        lags = [lag for _, lag, _ in shown_states]
        assert len(lags) > 50 and all(lag is not None and lag >= 0 for lag in lags), lags[:5]
        lead = [(generated - sound) / 48000 for sound, _, generated in shown_states]
        sync = f'sync: {len(lags)} states, lag median {1000 * np.median(lags) / 48000:.1f} ms, ' \
               f'max {1000 * max(lags) / 48000:.1f} ms, generation ahead by {np.median(lead):.2f}s median; '
        assert np.median(lags) / 48000 < 0.05 and np.median(lead) > 0.1
    else:
        sync = 'sync not checked (no audio device); '
    versions = sorted({s.config_version for s in snaps})
    lead = [s.timing['generated_seconds'] - s.timing['played_seconds'] for s in snaps
            if s.timing.get('played_seconds') is not None]
    print(f'GUI activity check passed: {path}; {sync}{len(snaps)} snapshots, rate {rate:.2f}/s, '
          f'dropped {w.worker.activity_slot.dropped}, config versions {versions[0]}..{versions[-1]}, '
          f'Fly OFF snapshots {len(off)}, manual journal {len(journal)} events, hold frames '
          f'{ends[0]["frames_applied"]}+{ends[1]["frames_applied"]}, final RTF {trace[-1]["rtf"]:.3f}, '
          f'underruns {trace[-1]["underruns"]}, lead {np.median(lead) if lead else float("nan"):.2f}s median',
          flush=True)
    w.close()
    QTimer.singleShot(500, app.quit)


w.worker.loaded.connect(loaded)
w.worker.telemetry.connect(observe)
w.worker.recording.connect(recorded)
w.resize(1500, 1000)
w.show()
QTimer.singleShot(100, w.load_model)
QTimer.singleShot(240000, lambda: (errors.append('timeout'), w.close(), app.quit()))
app.exec()
assert not errors, errors
assert stage['n'] == 10, stage
