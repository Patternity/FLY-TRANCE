"""Manual benchmark: generation timing while the GUI is used, on a real display.

Synthetic events are produced by Python in the GUI thread, so they cost GIL time themselves;
'hoverdummy' sends the same events to a plain label to measure that cost.

FLY_MODEL_DIR=/path/to/small [FLY_TEST_DEVICE=cuda] .venv/bin/python tests/gui_interaction_bench.py [scenario ...]

Runs in a visible window and injects synthetic Qt mouse events (the OS cursor does not move).
The audio device is opened at volume 0, so queue level and underruns are real. Scenarios:
  off     activity panel hidden, cursor still
  still   activity panel shown, cursor still
  hover   cursor moving over the 3D brain and the activity map (tooltips)
  rotate  left-drag rotation of the 3D scene
  hover3d / hovermap / hoverdummy   hover over one target only
  manual  manual Hold drag on an activity cell
"""
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QSettings, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

out = Path(os.environ.get('FLY_TEST_OUTPUT', '/tmp/fly-trance-interaction'))
out.mkdir(parents=True, exist_ok=True)
QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(out / 'settings'))
# The original BrainView renders before its window is mapped, which can block on X11 while VTK
# waits for the window. For a baseline of the original code, renders are skipped until shown.
if os.environ.get('FLY_BENCH_BRAIN_VIEW'):
    # Baseline: substitute another brain_view.py (e.g. the released one) into the current app.
    import importlib.util
    spec = importlib.util.spec_from_file_location('fly_trance.brain_view', os.environ['FLY_BENCH_BRAIN_VIEW'])
    module = importlib.util.module_from_spec(spec)
    sys.modules['fly_trance.brain_view'] = module
    spec.loader.exec_module(module)
if os.environ.get('FLY_BENCH_GUARD_EARLY_RENDER') == '1':
    from pyvistaqt import QtInteractor
    _render = QtInteractor.render
    ALLOW = [False]
    QtInteractor.render = lambda self: _render(self) if ALLOW[0] else None
from fly_trance.gui import Observatory  # noqa: E402

if os.environ.get('FLY_BENCH_SWITCH'):
    sys.setswitchinterval(float(os.environ['FLY_BENCH_SWITCH']))
SECONDS = float(os.environ.get('FLY_BENCH_SECONDS', '12'))
scenarios = sys.argv[1:] or ['off', 'still', 'hover', 'rotate', 'still']
app = QApplication([])
w = Observatory(os.environ['FLY_MODEL_DIR'])
w.device.setCurrentText(os.environ.get('FLY_TEST_DEVICE', 'cuda'))
w.monitor.setChecked(os.environ.get('FLY_TEST_MONITOR', '1') == '1')
w.volume.setValue(0)
w.resize(1500, 920)
# Window flags must not change after construction: Qt would recreate the native window and
# VTK, already bound to it, would present nothing. A new window opens on top anyway.
w.show()
w.raise_()
w.activateWindow()
if os.environ.get('FLY_BENCH_GUARD_EARLY_RENDER') == '1':
    app.processEvents()
    ALLOW[0] = True
    w.brain.reset_camera()
steps, gui = [], []
results = {}
state = {'index': -1, 'started': None, 'phase': None, 'tick': 0, 'underruns0': 0}


def wrap_engine():
    engine = w.worker.engine
    original = engine.frame

    def timed():
        t0 = time.perf_counter()
        result = original()
        steps.append((time.monotonic(), time.perf_counter() - t0))
        return result
    engine.frame = timed


def wrap_gui(obj, name, label):
    original = getattr(obj, name)

    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            gui.append((time.monotonic(), label, time.perf_counter() - t0))
    setattr(obj, name, timed)


wrap_gui(w.brain, 'update_state', 'brain.update_state')
wrap_gui(w.activity.map, 'paintEvent', 'activity.paint')
wrap_gui(w, 'on_telemetry', 'on_telemetry')

brain_widget = w.brain.plotter.interactor if hasattr(w.brain.plotter, 'interactor') else w.brain
mouse = {'pos': None}


def drive():
    """Synthetic input for the current phase, ~120 events per second."""
    phase, t = state['phase'], state['tick'] * 0.008
    state['tick'] += 1
    if phase in ('hover', 'hover3d', 'hovermap', 'hoverdummy'):
        # 'hover' alternates between the 3D view and the activity map every two seconds;
        # 'hoverdummy' sends the same synthetic events to a plain label to measure the driver.
        target = {'hover3d': brain_widget, 'hovermap': w.activity.map, 'hoverdummy': w.status_label}.get(
            phase, brain_widget if int(t / 2) % 2 == 0 else w.activity.map)
        r = target.rect()
        x = int(r.width() * (0.5 + 0.35 * math.sin(t * 3.1)))
        y = int(r.height() * (0.5 + 0.35 * math.sin(t * 2.3)))
        QTest.mouseMove(target, QPoint(x, y))
    elif phase == 'rotate':
        r = brain_widget.rect()
        cx, cy = r.width() // 2, r.height() // 2
        pos = QPoint(int(cx + 150 * math.sin(t * 1.7)), int(cy + 80 * math.sin(t * 1.1)))
        if mouse['pos'] is None:
            QTest.mousePress(brain_widget, Qt.LeftButton, Qt.NoModifier, QPoint(cx, cy))
        QTest.mouseMove(brain_widget, pos)
        mouse['pos'] = pos
    elif phase == 'manual':
        # Manual Hold on one activity cell, dragged up and down: strength changes every event.
        m = w.activity.map
        if mouse.get('manual') is None:
            w.activity.manual_more.setChecked(True)
            w.activity.manual_mode.setChecked(True)
            w.activity.manual_kind.setCurrentIndex(1)
            mouse['manual'] = m._cell_rectf(3, 30).center().toPoint()
            QTest.mousePress(m, Qt.LeftButton, Qt.NoModifier, mouse['manual'])
        QTest.mouseMove(m, mouse['manual'] - QPoint(0, int(60 * math.sin(t * 2.0))))


driver = QTimer(); driver.setInterval(8); driver.timeout.connect(drive)


def summarize(name, t0, t1):
    s = np.array([d for t, d in steps if t0 <= t < t1]) * 1000
    g = [(label, d) for t, label, d in gui if t0 <= t < t1]
    per = {}
    for label, d in g:
        per.setdefault(label, []).append(d * 1000)
    sink = w.worker.sink
    row = {
        'frames': int(s.size), 'step_mean_ms': float(s.mean()), 'step_p95_ms': float(np.percentile(s, 95)),
        'step_max_ms': float(s.max()), 'slow_frames_gt80ms': int((s > 80).sum()),
        'rtf': float(s.mean() / 40),
        'underruns': (sink.underruns - state['underruns0']) if sink else None,
        'queue_s_end': sink.queued / 48000 if sink else None,
        '3d_renders_per_s': (w.brain.render_stats['count'] - state['renders0'][0]) / (t1 - t0 + 2.0)
            if hasattr(w.brain, 'render_stats') else None,
        '3d_render_ms': 1000 * (w.brain.render_stats['seconds'] - state['renders0'][1])
            / max(w.brain.render_stats['count'] - state['renders0'][0], 1) if hasattr(w.brain, 'render_stats') else None,
        'gui': {k: {'calls_per_s': len(v) / (t1 - t0), 'mean_ms': statistics.mean(v), 'max_ms': max(v),
                    'busy_pct': 100 * sum(v) / 1000 / (t1 - t0)} for k, v in per.items()},
    }
    results[name] = row
    print(f"{name:>7}: step mean {row['step_mean_ms']:.1f} ms p95 {row['step_p95_ms']:.1f} max {row['step_max_ms']:.0f} "
          f">80ms {row['slow_frames_gt80ms']} RTF {row['rtf']:.3f} underruns {row['underruns']} "
          f"queue {row['queue_s_end'] if row['queue_s_end'] is None else round(row['queue_s_end'], 2)}s | "
          + (f"3D renders {row['3d_renders_per_s']:.0f}/s {row['3d_render_ms']:.1f}ms | " if row['3d_renders_per_s'] is not None else '') +
          ' '.join(f"{k} {v['calls_per_s']:.0f}/s {v['mean_ms']:.1f}ms busy {v['busy_pct']:.0f}%"
                   for k, v in row['gui'].items()), flush=True)


def next_phase():
    if state['phase'] is not None:
        driver.stop()
        if mouse['pos'] is not None:
            QTest.mouseRelease(brain_widget, Qt.LeftButton, Qt.NoModifier, mouse['pos'])
            mouse['pos'] = None
        if mouse.get('manual') is not None:
            QTest.mouseRelease(w.activity.map, Qt.LeftButton, Qt.NoModifier, mouse['manual'])
            mouse['manual'] = None
        name = f"{state['index']}:{state['phase']}"
        summarize(name, state['started'] + 2.0, time.monotonic())  # skip the transition
    state['index'] += 1
    if state['index'] >= len(scenarios):
        (out / 'interaction-bench.json').write_text(json.dumps(results, indent=2))
        w.worker.send('stop')
        QTimer.singleShot(1000, lambda: (w.close(), QTimer.singleShot(500, app.quit)))
        return
    phase = scenarios[state['index']]
    print('phase', phase, 'frames so far', len(steps), flush=True)
    state.update(phase=phase, started=time.monotonic(), tick=0)
    state['underruns0'] = w.worker.sink.underruns if w.worker.sink else 0
    stats = getattr(w.brain, 'render_stats', {'count': 0, 'seconds': 0.0})
    state['renders0'] = (stats['count'], stats['seconds'])
    w.show_activity.setChecked(phase != 'off')
    if phase in ('hover', 'hover3d', 'hovermap', 'hoverdummy', 'rotate', 'manual'):
        driver.start()
    QTimer.singleShot(int(SECONDS * 1000), next_phase)


def loaded(_):
    print('model loaded', flush=True)
    wrap_engine()
    w.fly.setChecked(True)
    w.start_stream()
    QTimer.singleShot(8000, next_phase)  # let warm-up and queue fill settle


w.worker.loaded.connect(loaded)
w.worker.error.connect(lambda e: print('worker error:', e, flush=True))
w.worker.status.connect(lambda s: print('status:', s, flush=True))
QTimer.singleShot(200, w.load_model)
QTimer.singleShot(int((len(scenarios) * SECONDS + 90) * 1000), lambda: (print('timeout'), app.quit()))
app.exec()
