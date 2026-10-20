"""Manual headless end-to-end UI/worker test with actual local weights."""
import json
import os
from pathlib import Path
import sys
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication
from fly_trance.gui import Observatory

# Keep the user's saved window geometry untouched.
QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, os.environ.get('FLY_TEST_OUTPUT', '/tmp/fly-trance-ui-test') + '/settings')
app = QApplication([])
w = Observatory(os.environ['FLY_MODEL_DIR'])
w.monitor.setChecked(False)
w.record.setChecked(True)
w.device.setCurrentText('cpu')
w.outdir.setText(os.environ.get('FLY_TEST_OUTPUT', '/tmp/fly-trance-ui-test'))
errors = []
stage = [0]
w.worker.error.connect(errors.append)

def loaded(_):
    w.fly.setChecked(True)
    w.start_stream()

def observe(data):
    if data['frame'] >= 4 and stage[0] == 0:
        w.fly.setChecked(False)
        stage[0] = 1
    elif data['frame'] >= 8 and stage[0] == 1:
        w.fly.setChecked(True)
        stage[0] = 2
    elif data['frame'] >= 12 and stage[0] == 2:
        stage[0] = 3
        w.worker.send('stop')

def recorded(path):
    trace = [json.loads(line) for line in Path(path).with_suffix('.telemetry.jsonl').read_text().splitlines()]
    assert any(r['delta_rms'] > 0 for r in trace)
    assert any(not r['enabled'] and r['delta_rms'] == 0 for r in trace)
    assert all(r['audio_sample_stop'] >= r['audio_sample_start'] for r in trace)
    w.grab().save('/tmp/fly-trance-ui-tested.png')
    print('GUI worker recording passed:', path, 'snapshots:', len(trace), flush=True)
    w.close()
    QTimer.singleShot(500, app.quit)

w.worker.loaded.connect(loaded)
w.worker.telemetry.connect(observe)
w.worker.recording.connect(recorded)
w.show()
QTimer.singleShot(100, w.load_model)
QTimer.singleShot(90000, lambda: (errors.append('timeout'), w.close(), app.quit()))
app.exec()
assert not errors, errors
assert stage[0] == 3
