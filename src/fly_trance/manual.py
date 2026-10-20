"""Manual interventions on temporal-layer activations: pulses, holds and block output bypass.

GUI thread -> ManualMailbox: thread-safe desired state. Drag updates overwrite the hold strength
(latest value wins), release and clear change the state directly so they cannot be lost behind
queued mouse events, and one-frame pulses wait in a bounded queue.

Worker thread -> ManualControl.begin_frame() at every model frame boundary turns the desired
state into the operations applied in that frame, and records journal events for what was
actually applied, with the frame number.

Semantics of an addition on layer i, coordinate group g (coordinates [start, stop)):
    ref = RMS over all coordinates of the layer vector immediately before the manual addition
          (after any bypass selection and CX512 addition on that layer);
          ref = REF_RMS_FLOOR when the RMS is smaller or not finite
    h[start:stop] += signed_strength * ref        (every coordinate of the group, nothing else)
Several operations on the same layer add up. Nothing is renormalized afterwards.
"""
import itertools
import math
import threading
import time
from dataclasses import dataclass, replace

import torch

from .activations import grouping

MAX_STRENGTH = 4.0          # |signed strength| limit, in units of the reference RMS
REF_RMS_FLOOR = 1e-3        # reference used when the vector RMS is below it or not finite
HOLD_TIMEOUT = 1.0          # seconds without a GUI refresh before a hold is released
MAX_PENDING_PULSES = 8
UPDATE_LOG_INTERVAL = 0.25  # seconds between journal entries for strength changes of one hold


def coordinates(group, group_size, dims):
    start = group * group_size
    return start, min(start + group_size, dims)


@dataclass(frozen=True)
class Hold:
    id: int
    layer: int
    group: int
    strength: float
    refreshed: float
    applied: bool = False
    release_after_apply: bool = False


@dataclass(frozen=True)
class Pulse:
    id: int
    layer: int
    group: int
    strength: float


class ManualMailbox:
    """Desired manual state, written by the GUI and read by the worker at frame boundaries."""

    def __init__(self, clock=time.monotonic):
        self._lock = threading.Lock()
        self.clock = clock
        self.layers = self.dims = None
        self._ids = itertools.count(1)
        self._hold = None
        self._pulses = []
        self._bypass = frozenset()
        self._clear = None  # pending clear request: number of pulses it dropped
        self._reasons = {}  # hold id -> why the GUI released it, for the journal

    def configure(self, layers, dims):
        with self._lock:
            self.layers, self.dims = layers, dims
            self._bypass = frozenset(i for i in self._bypass if i < layers)

    def _check(self, layer, group, strength):
        if self.layers is None:
            raise RuntimeError("Load a model before manual interventions")
        if not 0 <= layer < self.layers or not 0 <= group < grouping(self.dims)[1]:
            raise ValueError("Manual target is outside the model")
        self._check_strength(strength)

    @staticmethod
    def _check_strength(strength):
        if not math.isfinite(strength) or abs(strength) > MAX_STRENGTH:
            raise ValueError(f"Manual strength must be finite and within ±{MAX_STRENGTH}")

    def pulse(self, layer, group, strength):
        """Queue a one-frame addition; returns its id, or None when the bounded queue is full."""
        with self._lock:
            self._check(layer, group, strength)
            if len(self._pulses) >= MAX_PENDING_PULSES:
                return None
            pulse = Pulse(next(self._ids), layer, group, float(strength))
            self._pulses.append(pulse)
            return pulse.id

    def start_hold(self, layer, group, strength):
        with self._lock:
            self._check(layer, group, strength)
            hold = Hold(next(self._ids), layer, group, float(strength), self.clock())
            self._hold = hold
            return hold.id

    def update_hold(self, hold_id, strength=None):
        """Latest value wins; every call also refreshes the stale-hold watchdog."""
        with self._lock:
            if self._hold is None or self._hold.id != hold_id:
                return False
            if strength is not None:
                self._check_strength(strength)
            self._hold = replace(self._hold, refreshed=self.clock(),
                                 strength=self._hold.strength if strength is None else float(strength))
            return True

    def release(self, hold_id=None, reason="release"):
        with self._lock:
            hold = self._hold
            if hold is None or (hold_id is not None and hold.id != hold_id):
                return
            self._reasons = {hold.id: reason}
            # A hold always covers at least one model frame, even for a click shorter than a frame.
            self._hold = None if hold.applied else replace(hold, release_after_apply=True)

    def set_bypass(self, layer, enabled):
        with self._lock:
            if self.layers is not None and not 0 <= layer < self.layers:
                raise ValueError("Bypass layer is outside the model")
            self._bypass = self._bypass | {layer} if enabled else self._bypass - {layer}

    def bypass(self):
        with self._lock:
            return self._bypass

    def clear(self):
        """Drop holds and pending pulses and return every block from bypass."""
        with self._lock:
            dropped = len(self._pulses)
            self._pulses.clear()
            self._hold = None
            self._bypass = frozenset()
            self._clear = (self._clear or 0) + dropped

    def reset_session(self):
        """Stop or new session: no hold or pulse may continue into the next stream."""
        with self._lock:
            self._hold = None
            self._pulses.clear()

    def release_reason(self, hold_id):
        with self._lock:
            return self._reasons.get(hold_id, "release")

    def expire(self, hold_id):
        with self._lock:
            if self._hold is not None and self._hold.id == hold_id:
                self._hold = None

    def take(self):
        """Worker side, once per frame: (hold, pulses, bypass, clear) to apply now."""
        with self._lock:
            hold = self._hold
            if hold is not None:
                self._hold = None if hold.release_after_apply else replace(hold, applied=True)
            pulses, self._pulses = tuple(self._pulses), []
            clear, self._clear = self._clear, None
            return hold, pulses, self._bypass, clear

    def pending(self):
        with self._lock:
            return len(self._pulses), self._hold is not None


@dataclass
class FramePlan:
    frame: int
    bypass: frozenset
    ops: dict      # layer -> tuple of (start, stop, strength, kind, id)
    events: list   # journal events; "_ref_layer" marks those that need the reference RMS
    refs: dict     # layer -> reference RMS tensor, filled while the layers run
    version: int


class ManualControl:
    """Worker-side applied state; GUI code never touches it."""

    def __init__(self, layers, dims, clock=time.monotonic):
        self.layers, self.dims = layers, dims
        self.group_size, self.groups = grouping(dims)
        self.clock = clock
        self.mailbox = None
        self.version = 0
        self._cache = {}
        self._pending = []
        self.reset_session()

    def attach(self, mailbox):
        self.mailbox = mailbox
        mailbox.configure(self.layers, self.dims)

    def reset_session(self):
        self.hold = None       # dict for the applied hold, or None
        self.bypass = frozenset()
        self._pending = []
        self._cache.clear()

    def _coords(self, layer, group):
        return list(coordinates(group, self.group_size, self.dims))

    def _end_hold(self, frame, reason, now):
        hold, self.hold = self.hold, None
        return {"type": "manual_hold_end", "id": hold["id"], "effective_frame": frame,
                "last_applied_frame": frame - 1, "frames_applied": frame - hold["start_frame"],
                "layer": hold["layer"], "coordinates": self._coords(hold["layer"], hold["group"]),
                "final_strength": hold["strength"], "reason": reason, "monotonic": now}

    def end_session(self, frame, reason):
        """Journal events for state that ends with the stream (e.g. Stop)."""
        now = self.clock()
        events = [self._end_hold(frame, reason, now)] if self.hold is not None else []
        self.reset_session()
        return events

    def begin_frame(self, frame):
        """Read the desired state once per frame. None when nothing manual is active."""
        if self.mailbox is None:
            return None
        hold, pulses, bypass, clear = self.mailbox.take()
        if hold is None and not pulses and clear is None and self.hold is None and bypass == self.bypass:
            if not bypass:
                return None
            return FramePlan(frame, bypass, {}, [], {}, self.version)
        now = self.clock()
        events = []
        if clear is not None:
            released = self.hold["id"] if self.hold is not None else None
            if self.hold is not None:
                events.append(self._end_hold(frame, "clear", now))
            events.append({"type": "manual_clear", "effective_frame": frame, "released_hold": released,
                           "dropped_pulses": clear, "bypass_cleared": sorted(self.bypass - bypass),
                           "monotonic": now})
        if hold is not None and now - hold.refreshed > HOLD_TIMEOUT and not hold.release_after_apply:
            # The GUI stopped refreshing a pressed hold (e.g. it froze or lost the release).
            self.mailbox.expire(hold.id)
            if self.hold is not None and self.hold["id"] == hold.id:
                events.append(self._end_hold(frame, "timeout", now))
            hold = None
        if self.hold is not None and (hold is None or hold.id != self.hold["id"]):
            events.append(self._end_hold(frame, self.mailbox.release_reason(self.hold["id"]), now))
        if hold is not None:
            coords = self._coords(hold.layer, hold.group)
            if self.hold is None:
                self.hold = {"id": hold.id, "layer": hold.layer, "group": hold.group, "strength": hold.strength,
                             "start_frame": frame, "logged_strength": hold.strength, "logged_at": now}
                events.append({"type": "manual_hold_start", "id": hold.id, "effective_frame": frame,
                               "layer": hold.layer, "coordinates": coords, "strength": hold.strength,
                               "_ref_layer": hold.layer, "monotonic": now})
            else:
                self.hold["strength"] = hold.strength
                if (hold.strength != self.hold["logged_strength"]
                        and now - self.hold["logged_at"] >= UPDATE_LOG_INTERVAL):
                    self.hold.update(logged_strength=hold.strength, logged_at=now)
                    events.append({"type": "manual_hold_update", "id": hold.id, "effective_frame": frame,
                                   "layer": hold.layer, "coordinates": coords, "strength": hold.strength,
                                   "_ref_layer": hold.layer, "monotonic": now})
        for layer in sorted(bypass ^ self.bypass):
            if clear is None or layer in bypass:
                events.append({"type": "bypass", "effective_frame": frame, "layer": layer,
                               "enabled": layer in bypass, "monotonic": now})
        self.bypass = bypass
        ops = {}
        if self.hold is not None:
            h = self.hold
            ops.setdefault(h["layer"], []).append((*self._coords(h["layer"], h["group"]), h["strength"], "hold", h["id"]))
        for pulse in pulses:
            coords = self._coords(pulse.layer, pulse.group)
            ops.setdefault(pulse.layer, []).append((*coords, pulse.strength, "pulse", pulse.id))
            events.append({"type": "manual_pulse", "id": pulse.id, "effective_frame": frame, "layer": pulse.layer,
                           "coordinates": coords, "strength": pulse.strength, "_ref_layer": pulse.layer,
                           "monotonic": now})
        if events:
            self.version += 1
        return FramePlan(frame, bypass, {k: tuple(v) for k, v in ops.items()}, events, {}, self.version)

    def _tensors(self, layer, ops, device):
        key = (str(device), layer, tuple((a, b, s) for a, b, s, _, _ in ops))
        cached = self._cache.get(key)
        if cached is None:
            if len(self._cache) > 64:
                self._cache.clear()
            # Built on the model device from Python scalars: no host-to-device copy of a vector.
            idx = torch.cat([torch.arange(a, b, device=device) for a, b, _, _, _ in ops])
            values = torch.cat([torch.full((b - a,), float(s), device=device) for a, b, s, _, _ in ops])
            cached = self._cache[key] = (idx, values)
        return cached

    def apply(self, plan, layer, h):
        """Step 6 of the block order: add the manual vector to the selected coordinates only."""
        ops = plan.ops.get(layer)
        if not ops:
            return h
        idx, values = self._tensors(layer, ops, h.device)
        rms = h.detach().float().square().mean().sqrt()
        # Bounded, finite reference without a device sync: non-finite or tiny RMS -> floor.
        ref = torch.nan_to_num(rms, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(REF_RMS_FLOOR)
        plan.refs[layer] = ref
        add = (values * ref).to(h.dtype).view(*([1] * (h.dim() - 1)), -1)
        return h.index_add(h.dim() - 1, idx, add)

    def finish_frame(self, plan):
        for event in plan.events:
            layer = event.pop("_ref_layer", None)
            if layer is not None:
                event["_ref"] = plan.refs.get(layer)
        self._pending.extend(plan.events)

    def resolve(self):
        """After the frame's own device sync: attach reference RMS values to journal events."""
        events, self._pending = self._pending, []
        pairs = [(event, event.pop("_ref", None)) for event in events]
        tensors = [ref for _, ref in pairs if ref is not None]
        # One small transfer for all events of the frame; the stream is already synchronized.
        values = iter(torch.stack(tensors).float().cpu().tolist() if tensors else ())
        for event, ref in pairs:
            if ref is not None:
                value = next(values)
                event["reference_rms"] = value
                event["per_coordinate"] = event["strength"] * value
        return events
