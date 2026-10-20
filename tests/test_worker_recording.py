"""The real Qt worker writes session files only when recording is requested."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication

from fly_trance.worker import MusicWorker


class FakeEngine:
    def __init__(self, *args):
        self.device = "cpu"
        self.coupling = SimpleNamespace(probe=SimpleNamespace(layers=12, dims=1024),
                                        manual=SimpleNamespace(attach=lambda mailbox: None))
        self.count = 0

    def set_observe(self, enabled):
        pass

    def reset(self, seed):
        self.count = 0

    def configure(self, config):
        return {}

    def frame(self):
        self.count += 1
        return np.zeros((1920, 2), np.float32), {"frame": self.count, "delta_rms": 0.0, "state": np.zeros(512)}

    def take_activations(self):
        return None

    def take_events(self):
        return []

    def end_manual(self, reason):
        return []

    def metadata(self):
        return {"fake": True}


def wait_for(condition, seconds=5):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert condition()


def session(folder, record):
    QCoreApplication.instance() or QCoreApplication([])
    worker = MusicWorker(engine_factory=FakeEngine)
    worker.start()
    try:
        worker.send("load", model="fake", device="cpu")
        worker.send("start", seed=1, config=None, output=str(folder / "out" / "session.wav"),
                    monitor=False, record=record)
        wait_for(lambda: worker.frames >= 30)
        worker.send("stop")
        wait_for(lambda: not worker.running)
    finally:
        worker.shutdown.set()
        worker.wait(5000)


def test_recording_off_writes_nothing(tmp_path):
    session(tmp_path, record=False)
    assert list(tmp_path.iterdir()) == []


def test_recording_on_writes_wav_metadata_and_telemetry(tmp_path):
    session(tmp_path, record=True)
    out = tmp_path / "out"
    assert {p.name for p in out.iterdir()} == {"session.wav", "session.json", "session.telemetry.jsonl"}
    import soundfile as sf
    assert sf.info(str(out / "session.wav")).frames % 1920 == 0 and sf.info(str(out / "session.wav")).frames > 0
