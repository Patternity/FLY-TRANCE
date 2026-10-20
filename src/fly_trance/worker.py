"""One model thread, serial commands, frame-boundary Fly changes and WAV capture."""
import gc
import json
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QThread, Signal

from .activations import LatestSlot
from .audio import AudioOutput
from .manual import ManualMailbox


class MusicWorker(QThread):
    status = Signal(str)
    loaded = Signal(dict)
    # Emitted only when the previous record was taken from telemetry_slot, so a slow GUI never
    # builds a backlog; the slot always holds the newest record.
    telemetry = Signal(dict)
    event = Signal(dict)
    error = Signal(str)
    recording = Signal(str)
    # Payload-free wake-up; the snapshot itself waits in activity_slot.
    activations = Signal()

    def __init__(self, parent=None, engine_factory=None):
        super().__init__(parent)
        self.commands = queue.Queue()
        self.shutdown = threading.Event()
        self.engine_factory = engine_factory
        self.engine = None
        self.running = False
        self.sink = None
        self.writer = None
        self.output = None
        self.trace_writer = None
        self.volume = 0.75
        self.generated_samples = 0
        self.frames = 0
        self.last_emit = 0.0
        self.observe = False
        self.activity_slot = LatestSlot()
        self.telemetry_slot = LatestSlot()
        # Thread-safe desired manual state; the GUI writes it, frames read it.
        self.manual = ManualMailbox()
        self.step_times = deque(maxlen=125)  # about five seconds of model frames

    def send(self, name, **payload):
        self.commands.put((name, payload))

    def _stop(self):
        self.running = False
        # Holds and pending pulses never continue into another session.
        self.manual.reset_session()
        if self.engine is not None:
            for event in self.engine.end_manual("stop"):
                self.event.emit(event)
        if self.sink:
            self.sink.close()
            self.sink = None
        if self.trace_writer:
            self.trace_writer.close()
            self.trace_writer = None
        if self.writer:
            self.writer.close()
            self.writer = None
            try:
                self.output.with_suffix(".json").write_text(json.dumps(self.engine.metadata(), indent=2))
                self.recording.emit(str(self.output))
            except Exception as exc:
                self.error.emit(f"Recording metadata failed: {exc}")
        self.status.emit("Ready" if self.engine else "No model loaded")

    def _command(self, name, data):
        if name == "load":
            if self.running:
                raise RuntimeError("Stop the stream before changing the model")
            self.status.emit("Loading model; first use downloads the pinned checkpoint...")
            if self.engine_factory is None:
                from .engine import MusicEngine
                self.engine_factory = MusicEngine
            self.engine = None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.engine = self.engine_factory(data["model"], data["device"], data.get("style", ""))
            self.engine.set_observe(self.observe)
            self.engine.coupling.manual.attach(self.manual)
            probe = self.engine.coupling.probe
            self.loaded.emit({"device": self.engine.device, "layers": probe.layers, "dims": probe.dims})
            self.status.emit("Ready")
        elif name == "start":
            if self.engine is None or self.running:
                raise RuntimeError("Load a model before starting a new session")
            self.engine.reset(data["seed"])
            self.engine.configure(data["config"])
            # Without record nothing is written to disk: no WAV, metadata or telemetry.
            self.output = Path(data["output"]) if data.get("record", False) else None
            try:
                if self.output is not None:
                    self.output.parent.mkdir(parents=True, exist_ok=True)
                    self.writer = sf.SoundFile(self.output, "w", samplerate=48000, channels=2, subtype="PCM_16")
                    self.trace_writer = self.output.with_suffix(".telemetry.jsonl").open("w")
                if data.get("monitor", True):
                    self.sink = AudioOutput()
                    self.sink.volume = self.volume
                    self.sink.start()
            except Exception:
                self._stop()
                raise
            self.generated_samples = 0
            self.frames = 0
            self.step_times.clear()
            self.running = True
            self.started_at = time.monotonic()
            self.status.emit("Streaming")
        elif name == "stop":
            self._stop()
        elif name == "fly":
            if self.engine:
                self.event.emit(self.engine.configure(data["config"]))
        elif name == "observe":
            # Only observation changes; stream, cache, codec and RNG are untouched.
            self.observe = bool(data["enabled"])
            if self.engine:
                self.engine.set_observe(self.observe)
        elif name == "volume":
            self.volume = data["value"]
            if self.sink:
                self.sink.volume = self.volume
        elif name == "style":
            if self.engine:
                self.status.emit("Preparing style conditioning...")
                self.engine.set_style(data["style"])
                event = {"type": "style", "style": data["style"], "effective_frame": self.engine.coupling.frame}
                self.engine.events.append(event)
                self.event.emit(event)
                self.status.emit("Streaming" if self.running else "Ready")

    def audible_sample(self):
        """Generated sample index heard now, or None without an audio device. Safe from the GUI thread."""
        sink = self.sink
        return None if sink is None else sink.audible()

    def _step_stats(self):
        """Model step timing over the last ~5 s; cheap enough for every telemetry record."""
        steps = np.fromiter(self.step_times, dtype=float) * 1000
        if not steps.size:
            return {}
        return {"step_ms_mean": float(steps.mean()), "step_ms_p95": float(np.percentile(steps, 95)),
                "step_ms_max": float(steps.max()), "slow_steps": int((steps > 60).sum())}

    def _deliver_activations(self):
        snapshot = self.engine.take_activations()
        if snapshot is None or not self.running:
            return
        sink = self.sink
        snapshot.timing = {"generated_seconds": self.generated_samples / 48000,
                           "sound_sample": self.generated_samples,
                           "played_seconds": sink.played / 48000 if sink else None,
                           "queue_seconds": sink.queued / 48000 if sink else 0.0}
        # Never waits for the GUI: an undelivered older snapshot is replaced.
        if self.activity_slot.put(snapshot):
            self.activations.emit()

    def run(self):
        try:
            while not self.shutdown.is_set():
                try:
                    name, data = self.commands.get(timeout=0.02 if not self.running else 0)
                except queue.Empty:
                    name = None
                if name is not None:
                    try:
                        self._command(name, data)
                    except Exception as exc:
                        self.error.emit(str(exc))
                        if name == "load":
                            self.status.emit("Model loading failed")
                        elif name == "start":
                            self._stop()
                        elif name == "style":
                            self.status.emit("Streaming" if self.running else "Ready")
                    # Drain commands before the next frame so off is not delayed by telemetry.
                    continue
                if not self.running:
                    continue
                try:
                    started = time.perf_counter()
                    audio, snapshot = self.engine.frame()
                    self.step_times.append(time.perf_counter() - started)
                    # Confirmations come from the frame that applied them, with its frame number.
                    for event in self.engine.take_events():
                        self.event.emit(event)
                    audio_sample_start = self.generated_samples
                    if self.writer:
                        self.writer.write(audio)
                    self.generated_samples += len(audio)
                    if self.sink:
                        while not self.sink.push(audio) and not self.shutdown.is_set():
                            # A bounded queue applies backpressure. Commands are handled while waiting.
                            try:
                                name, data = self.commands.get(timeout=0.01)
                                self._command(name, data)
                            except queue.Empty:
                                pass
                            if not self.running or self.sink is None:
                                break
                    self.frames += 1
                    if self.observe:
                        self._deliver_activations()
                    now = time.monotonic()
                    if now - self.last_emit >= 0.1:
                        self.last_emit = now
                        # The codec answers with one frame of latency: the sound of this frame's tokens
                        # starts with the next chunk, i.e. at sample audio_sample_stop.
                        record = {**snapshot, "audio_sample_start": audio_sample_start, "audio_sample_stop": self.generated_samples,
                            "sound_sample": self.generated_samples,
                            "generated_seconds": self.generated_samples / 48000,
                            "played_seconds": self.sink.played / 48000 if self.sink else None,
                            "queue_seconds": self.sink.queued / 48000 if self.sink else 0,
                            "underruns": self.sink.underruns if self.sink else 0,
                            "rtf": (now-self.started_at) / max(self.generated_samples / 48000, 0.04),
                            **self._step_stats()}
                        if self.trace_writer:
                            self.trace_writer.write(json.dumps({**record, "state": record["state"].tolist()}) + "\n")
                        if self.telemetry_slot.put(record):
                            self.telemetry.emit(record)
                except Exception as exc:
                    self._stop()
                    self.error.emit(f"Generation stopped: {exc}")
        finally:
            self._stop()
