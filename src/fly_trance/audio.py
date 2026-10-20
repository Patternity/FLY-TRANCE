"""Bounded callback playback; the audio callback never performs model work."""
from collections import deque
import threading

import numpy as np


class AudioOutput:
    def __init__(self, sample_rate=48000, capacity_seconds=1.0):
        self.sample_rate = sample_rate
        self.capacity = round(sample_rate * capacity_seconds)
        self.chunks = deque()
        self.queued = 0
        self.played = 0
        self.underruns = 0
        self.volume = 0.75
        self.lock = threading.Lock()
        self.stream = None
        self.latency_samples = 0

    def start(self):
        import sounddevice as sd
        self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=2,
                                      dtype="float32", blocksize=960, callback=self._callback)
        self.stream.start()
        # Output latency reported by PortAudio: samples handed over now are heard this much later.
        self.latency_samples = round(float(self.stream.latency) * self.sample_rate)

    def audible(self):
        """Index of the generated sample being heard now (an estimate based on PortAudio's latency)."""
        return self.played - self.latency_samples

    def _callback(self, out, frames, timing, status):
        out.fill(0)
        count = 0
        with self.lock:
            while count < frames and self.chunks:
                head = self.chunks[0]
                n = min(frames-count, len(head))
                out[count:count+n] = head[:n] * self.volume
                if n == len(head):
                    self.chunks.popleft()
                else:
                    self.chunks[0] = head[n:]
                count += n
                self.queued -= n
            self.played += count
            if count < frames:
                self.underruns += 1

    def push(self, chunk):
        if not len(chunk):
            return True
        with self.lock:
            if self.queued + len(chunk) > self.capacity:
                return False
            self.chunks.append(chunk.copy())
            self.queued += len(chunk)
        return True

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        with self.lock:
            self.chunks.clear()
            self.queued = 0
