"""Show generated states when their sound is heard, instead of when they were computed.

Generation runs ahead of the speakers by the playback queue. Each state carries the index of the
first output sample of its own sound (the codec answers with one model frame of latency, so the
sound of frame t starts with the next audio chunk). The audio device reports which generated sample
is heard now: samples handed to it minus PortAudio's output latency. A state is shown once its sound
has started; if several became due at once, only the newest of each kind is shown.
"""
import bisect
import itertools

from PySide6.QtCore import QObject, Qt, QTimer

SAMPLE_RATE = 48000


class PlaybackAligner(QObject):
    """Bounded buffer of states, released at the moment their sound becomes audible.

    audible() -> generated sample index heard now, or None without an audio device.
    release(kind, payload, lag_samples) shows a state; lag_samples is how long after the start of
    its sound it is shown, or None when shown at generation time (no audio clock, or sync off).
    """

    def __init__(self, audible, release, capacity=128, parent=None):
        super().__init__(parent)
        self.audible, self.release = audible, release
        self.capacity = capacity
        self.enabled = True
        self.items = []  # sorted (sound_sample, sequence, kind, payload)
        self._sequence = itertools.count()
        # One precise single-shot timer armed for the next due state; no polling.
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.flush)

    def push(self, sound_sample, kind, payload):
        now = self.audible() if self.enabled and sound_sample is not None else None
        if now is None:
            self.release(kind, payload, None)
            return
        bisect.insort(self.items, (sound_sample, next(self._sequence), kind, payload), key=lambda item: item[:2])
        if len(self.items) > self.capacity:
            del self.items[0]  # bounded: the oldest waiting state is skipped
        self.flush()

    def flush(self):
        now = self.audible()
        if now is None:
            # The audio device is gone (Stop): this sound will never be heard.
            self.clear()
            return
        due = 0
        while due < len(self.items) and self.items[due][0] <= now:
            due += 1
        newest = {}
        for sound_sample, _, kind, payload in self.items[:due]:
            newest[kind] = (sound_sample, payload)
        del self.items[:due]
        for kind, (sound_sample, payload) in newest.items():
            self.release(kind, payload, now - sound_sample)
        if self.items:
            wait_ms = (self.items[0][0] - now) * 1000 / SAMPLE_RATE
            # Re-checked on firing, so a stalled device (underrun) simply keeps the state waiting.
            self.timer.start(max(1, min(int(wait_ms) + 1, 1000)))
        else:
            self.timer.stop()

    def clear(self):
        self.items.clear()
        self.timer.stop()

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        if not self.enabled:
            newest = {kind: payload for _, _, kind, payload in self.items}
            self.clear()
            for kind, payload in newest.items():
                self.release(kind, payload, None)
