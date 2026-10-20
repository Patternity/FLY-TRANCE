"""Explicit taps in MRT2's functional temporal step, independent of forward hooks."""
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

from .activations import ActivationProbe
from .manual import ManualControl
from .reservoir import Reservoir


@dataclass(frozen=True)
class FlyConfig:
    enabled: bool = False
    strength: float = 0.15
    leak: float = 0.2
    capture: int = 5
    inject: int = 11
    ensemble: str = "all"
    gain: float = 0.9
    input_strength: float = 0.5

    def validate(self, layers=12):
        if not np.isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError("Strength must be in [0, 1]")
        if not np.isfinite(self.leak) or not 0 < self.leak <= 1:
            raise ValueError("Leak must be in (0, 1]")
        if not 0 <= self.capture < layers or not 0 <= self.inject < layers:
            raise ValueError("Temporal layer is outside the model")
        if not np.isfinite(self.gain) or not 0 <= self.gain < 1:
            raise ValueError("Recurrent gain must be in [0, 1)")
        if not np.isfinite(self.input_strength) or not 0 <= self.input_strength <= 3:
            raise ValueError("Input strength must be in [0, 3]")
        if self.ensemble not in ("all", "DA", "ACH", "GABA", "GLUT", "SER"):
            raise ValueError("Unknown output ensemble")


class FlyCoupling:
    def __init__(self, decoder, graph, nodes, seed=42):
        self.decoder = decoder
        self.brain = Reservoir(graph, features=64, seed=seed)
        self.base_w = self.brain.w / self.brain.dynamics.gain
        width = decoder.cfg.temporal.model_dims
        rng = np.random.default_rng(seed)
        self.pin = rng.normal(0, 1 / np.sqrt(width), (width, 64)).astype(np.float32)
        self.pout = rng.normal(0, 1 / np.sqrt(len(nodes)), (len(nodes), width)).astype(np.float32)
        self.nodes = nodes
        self.config = FlyConfig(capture=min(5, len(decoder.temporal_body.layers)-1),
                                inject=len(decoder.temporal_body.layers)-1)
        self.frame = 0
        self.version = 0
        self.last = {}
        self.mask = np.ones(len(nodes))
        # Disabled by default; observation never changes values used by the model.
        self.probe = ActivationProbe(len(decoder.temporal_body.layers), width)
        # Inactive until a mailbox is attached; with nothing manual the path is unchanged.
        self.manual = ManualControl(len(decoder.temporal_body.layers), width)

    def configure(self, config):
        config.validate(len(self.decoder.temporal_body.layers))
        mask = np.array([config.ensemble == "all" or n["nt_type"] == config.ensemble
                         for n in self.nodes], dtype=float)
        if not mask.any():
            raise ValueError("Selected ensemble is empty")
        self.brain.set_leak(config.leak)
        self.brain.w = (self.base_w * config.gain).tocsr()
        self.brain.dynamics = replace(self.brain.dynamics, gain=config.gain, input_strength=config.input_strength)
        self.brain._update_fingerprint()
        self.config, self.mask = config, mask
        self.version += 1

    def _capture(self, h):
        # Batch 1 is intentional; external CFG batching is not used by this app.
        values = h[0, 0].detach().float().cpu().numpy() @ self.pin
        values /= max(float(np.sqrt(np.mean(values * values))), 1e-6)
        self.brain.advance(values)

    def _inject(self, h):
        if not self.config.enabled or self.config.strength == 0:
            return h, 0.0
        raw = (self.brain.state * self.mask) @ self.pout
        delta = torch.as_tensor(raw, dtype=h.dtype, device=h.device).view_as(h)
        # A fixed projection preserves reservoir amplitude; no per-frame output whitening.
        delta = delta * h.float().square().mean().sqrt().to(h.dtype) * self.config.strength
        if not torch.isfinite(delta).all():
            raise ValueError("Nonfinite Fly injection")
        return h + delta, float(delta.float().square().mean().sqrt())

    def temporal_step(self, prev_frame, self_kv, cross_kv, source_frame):
        stack = self.decoder.temporal_body
        h = self.decoder.embed(prev_frame).float().mean(-2).to(source_frame.dtype)
        next_self, next_cross = [], []
        delta_rms = 0.0
        # Chosen once per frame, so every column of a snapshot belongs to this frame.
        tap = self.probe.begin(self.frame)
        # Manual commands are read once per frame, at the frame boundary.
        plan = self.manual.begin_frame(self.frame)
        bypass = plan.bypass if plan is not None else ()
        # Preserve the upstream functional path; no hidden-state reset or cache flush.
        # Order inside every block:
        #   1 keep the block input   2 run the block and extend its caches as usual
        #   3 bypass: forward the kept input instead of the computed output
        #   4 CX512 read   5 CX512 addition (Fly ON)   6 manual addition   7 pass on
        for i, block in enumerate(stack.layers):
            x_in = h
            h, k, v = block["self_attention"].step_fn(h, *self_kv[i])
            next_self.append((k, v))
            if stack.use_cross:
                ca = block["cross_attention"]
                sk, sv = ca._kv(source_frame)
                sk = torch.cat([cross_kv[i][0], sk], 1)
                sv = torch.cat([cross_kv[i][1], sv], 1)
                next_cross.append((sk, sv))
                h = ca.attend_fn(h, sk, sv)
            h = block["ffn"](h)
            if i in bypass:
                # Block output bypass: the block still ran and its KV caches advanced above;
                # only its output is discarded. No computation is saved.
                h = x_in
            base = h
            if i == self.config.capture:
                self._capture(h)
            if i == self.config.inject:
                h, delta_rms = self._inject(h)
            mid = h
            if plan is not None:
                h = self.manual.apply(plan, i, h)
            if tap is not None:
                tap.layer(base, mid, h)
        if plan is not None:
            self.manual.finish_frame(plan)
        if tap is not None:
            meta = {"version": self.version, "config": asdict(self.config)}
            if plan is not None:
                meta.update(refs=plan.refs, bypass=sorted(plan.bypass), manual_version=plan.version,
                            manual_ops=tuple((layer, *op) for layer, ops in plan.ops.items() for op in ops))
            self.probe.finish(tap, meta)
        self.last = {
            "frame": self.frame, "audio_time": self.frame / 25,
            "enabled": self.config.enabled, "delta_rms": delta_rms,
            "state": self.brain.state.copy(), "version": self.version,
            "config": asdict(self.config),
            "delayed": self.config.inject < self.config.capture,
            "bypass": sorted(bypass), "manual_layers": sorted(plan.ops) if plan is not None else [],
        }
        self.frame += 1
        return h, next_self, next_cross
