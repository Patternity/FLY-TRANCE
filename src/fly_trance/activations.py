"""Read-only observation of temporal-layer activations on the executed functional path.

A frame is chosen once, before the layer loop. Only on that frame are the residual
activations of every temporal block summarized on the model device and sent to the CPU
as one packed transfer. Other frames do no extra work. Nothing here writes into tensors
used by the model.
"""
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

POSITION = ("residual stream after FFN (x + FFN(x)), or the forwarded block input under bypass, "
            "before the CX512 and manual additions")
PARTS = ("base", "cx512", "manual", "result")
BASE, CX512, MANUAL, RESULT = range(4)
FRAME_SECONDS = 0.04  # 1920 samples at 48 kHz per model frame
STATS = ("rms", "min", "max", "mean")


def grouping(dims, cells=64):
    """Contiguous, deterministic coordinate groups: (coordinates per group, group count)."""
    if dims < 1 or cells < 1:
        raise ValueError("Dimensions and cell count must be positive")
    size = math.ceil(dims / cells)
    return size, math.ceil(dims / size)


@dataclass
class ActivationSnapshot:
    frame: int
    music_time: float
    config_version: int
    config: dict
    layers: int
    dims: int
    group_size: int
    groups: int
    # [4, layers, groups] signed mean and RMS of every coordinate group, for each of PARTS:
    # base (position above), CX512 delta, manual delta, result passed to the next block.
    # Deltas are measured as differences in the model dtype, so they are exactly what was added.
    mean: np.ndarray
    rms: np.ndarray
    # [4, layers, 4] whole-vector statistics, ordered as STATS.
    stats: np.ndarray
    # [layers] reference RMS of the manual additions in this frame (0 where none).
    manual_ref: np.ndarray
    bypass: tuple = ()
    manual_ops: tuple = ()      # (layer, start, stop, strength, kind, id) applied in this frame
    manual_version: int = 0
    position: str = POSITION
    timing: dict = field(default_factory=dict)

    def delta_ratio(self, part, layer):
        """RMS of a delta relative to the vector it was added to."""
        reference = self.stats[BASE, layer, 0] if part == CX512 else self.manual_ref[layer]
        return float(self.stats[part, layer, 0] / reference) if reference > 0 else 0.0


class _Tap:
    """Collects tensor references for one frame; no kernels run until finish()."""

    def __init__(self, frame):
        self.frame = frame
        self.base, self.mid, self.after = [], [], []

    def layer(self, base, mid, after):
        # base: forwarded block output (or input under bypass); mid: after CX512; after: after manual.
        self.base.append(base)
        self.mid.append(mid)
        self.after.append(after)


class ActivationProbe:
    def __init__(self, layers, dims, cells=64, interval=0.2, clock=time.monotonic):
        self.layers, self.dims = layers, dims
        self.group_size, self.groups = grouping(dims, cells)
        self.interval = interval
        self.clock = clock
        self.enabled = False
        self.due = -math.inf
        self._counts = None
        self._host = None
        self._event = None
        self._pending = None

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        if not self.enabled:
            self.reset()

    def reset(self):
        self._pending = None
        self.due = -math.inf

    def begin(self, frame):
        """Decide once per frame, before the layer loop, whether this frame is observed."""
        if not self.enabled:
            return None
        now = self.clock()
        if now < self.due:
            return None
        # A fixed schedule keeps the average rate at 1/interval for any model frame rate;
        # after a long pause it restarts instead of bursting.
        self.due = self.due + self.interval if now - self.due < self.interval else now + self.interval
        return _Tap(frame)

    def _summaries(self, rows):
        # rows: [n, dims] float32 on the model device -> one flat tensor of all summaries.
        n = rows.shape[0]
        pad = self.group_size * self.groups - self.dims
        grouped = torch.nn.functional.pad(rows, (0, pad)).view(n, self.groups, self.group_size)
        if self._counts is None or self._counts.device != rows.device:
            counts = torch.full((self.groups,), float(self.group_size), device=rows.device)
            counts[-1] -= pad
            self._counts = counts
        mean = grouped.sum(-1) / self._counts
        rms = (grouped.square().sum(-1) / self._counts).sqrt()
        stats = torch.stack([rows.square().mean(-1).sqrt(), rows.amin(-1), rows.amax(-1), rows.mean(-1)], -1)
        return torch.cat([mean.flatten(), rms.flatten(), stats.flatten()])

    def finish(self, tap, meta):
        if tap is None:
            return
        if len(tap.base) != self.layers:
            raise RuntimeError("Activation tap did not see every temporal layer")
        refs = meta.pop("refs", {})
        with torch.no_grad():
            def stack(tensors):
                return torch.stack([h.detach().reshape(-1) for h in tensors]).float()
            base, mid, after = stack(tap.base), stack(tap.mid), stack(tap.after)
            # Measured additions in the model's own dtype; exactly zero where nothing was added.
            rows = torch.cat([base, mid - base, after - mid, after])
            ref = torch.zeros(self.layers, device=rows.device)
            for layer, value in refs.items():
                ref[layer] = value
            flat = torch.cat([self._summaries(rows), ref])
            if flat.device.type == "cuda":
                # One asynchronous packed copy; collect() runs after the frame's own audio sync.
                if self._host is None or self._host.numel() != flat.numel():
                    self._host = torch.empty(flat.numel(), dtype=torch.float32, pin_memory=True)
                    self._event = torch.cuda.Event()
                self._host.copy_(flat, non_blocking=True)
                self._event.record()
                payload = None
            else:
                payload = flat.cpu().numpy().copy()
        self._pending = (payload, {**meta, "frame": tap.frame})

    def collect(self):
        """Return the pending snapshot as numpy data, or None if this frame was not observed."""
        if self._pending is None:
            return None
        payload, meta = self._pending
        self._pending = None
        if payload is None:
            self._event.synchronize()
            payload = self._host.numpy().copy()
        L, g, k = self.layers, self.groups, len(PARTS)
        n = k * L
        mean = payload[:n * g].reshape(k, L, g)
        rms = payload[n * g:2 * n * g].reshape(k, L, g)
        stats = payload[2 * n * g:2 * n * g + n * len(STATS)].reshape(k, L, len(STATS))
        manual_ref = payload[2 * n * g + n * len(STATS):]
        return ActivationSnapshot(
            frame=meta["frame"], music_time=meta["frame"] * FRAME_SECONDS,
            config_version=meta["version"], config=meta["config"],
            layers=L, dims=self.dims, group_size=self.group_size, groups=g,
            mean=mean, rms=rms, stats=stats, manual_ref=manual_ref,
            bypass=tuple(meta.get("bypass", ())), manual_ops=tuple(meta.get("manual_ops", ())),
            manual_version=meta.get("manual_version", 0))


class LatestSlot:
    """Single-slot hand-off: a newer snapshot replaces an undelivered older one."""

    def __init__(self):
        self._lock = threading.Lock()
        self._item = None
        self.dropped = 0

    def put(self, item):
        """Store item; True when the slot was empty, i.e. the reader needs a wake-up."""
        with self._lock:
            empty = self._item is None
            if not empty:
                self.dropped += 1
            self._item = item
            return empty

    def take(self):
        with self._lock:
            item, self._item = self._item, None
            return item
