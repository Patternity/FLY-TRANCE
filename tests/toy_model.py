"""A small functional stack with MRT2's temporal-step interface, for tests without weights."""
import functools
import time
from dataclasses import replace
from types import SimpleNamespace

import torch

from fly_trance.coupling import FlyCoupling
from fly_trance.graph import load_graph
from fly_trance.manual import ManualMailbox

LAYERS, DIMS = 12, 130  # 130 needs a padded last group: 44 groups of 3, the last holds 1


class ToySelf:
    def __init__(self, d, g):
        self.w = torch.randn(d, d, generator=g) / d ** 0.5

    def step_fn(self, x, k_prev, v_prev):
        k = torch.cat([k_prev, x[:, :, None]], 1)
        v = torch.cat([v_prev, torch.tanh(x)[:, :, None]], 1)
        return x + torch.tanh(v.mean(1).squeeze(1)[:, None] @ self.w), k, v


class ToyCross:
    def _kv(self, source):
        return source[:, :, None], torch.sin(source)[:, :, None]

    def attend_fn(self, x, k, v):
        return x + 0.1 * torch.tanh(v.mean(1).squeeze(1)[:, None] + k.mean(1).squeeze(1)[:, None])


class ToyFFN(torch.nn.Module):
    def __init__(self, d, g):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(d, d, generator=g) / d ** 0.5, requires_grad=False)

    def forward(self, x):
        return x + torch.tanh(x @ self.w)


class ToyDecoder:
    """Same functional interface that FlyCoupling.temporal_step drives in MRT2."""

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.cfg = SimpleNamespace(temporal=SimpleNamespace(model_dims=DIMS))
        self.table = torch.randn(32, DIMS, generator=g)
        self.temporal_body = SimpleNamespace(use_cross=True, layers=[
            {"self_attention": ToySelf(DIMS, g), "cross_attention": ToyCross(), "ffn": ToyFFN(DIMS, g)}
            for _ in range(LAYERS)])

    def embed(self, tokens):
        return self.table[tokens]


def original_temporal_step(self, prev_frame, self_kv, cross_kv, source_frame):
    """FlyCoupling.temporal_step exactly as released in v0.1.0, before observation and manual control."""
    stack = self.decoder.temporal_body
    h = self.decoder.embed(prev_frame).float().mean(-2).to(source_frame.dtype)
    next_self, next_cross = [], []
    delta_rms = 0.0
    for i, block in enumerate(stack.layers):
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
        if i == self.config.capture:
            self._capture(h)
        if i == self.config.inject:
            h, delta_rms = self._inject(h)
    self.frame += 1
    return h, next_self, next_cross


@functools.lru_cache(maxsize=1)
def graph():
    return load_graph()


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class ToyStream:
    """Runs frames through FlyCoupling.temporal_step and records what the manual step saw."""

    def __init__(self, enabled=True, observe=True, capture=3, inject=8, strength=0.3, attach=True):
        g, nodes, _ = graph()
        self.decoder = ToyDecoder()
        self.fly = FlyCoupling(self.decoder, g, nodes)
        self.fly.configure(replace(self.fly.config, enabled=enabled, strength=strength, capture=capture,
                                   inject=inject))
        self.fly.probe.interval = 0
        self.fly.probe.set_enabled(observe)
        self.clock = Clock()
        self.mailbox = ManualMailbox(clock=self.clock)
        self.fly.manual.clock = self.clock
        if attach:
            self.fly.manual.attach(self.mailbox)
        z = torch.zeros(1, 0, 1, DIMS)
        self.self_kv = self.cross_kv = [(z, z)] * LAYERS
        self.prev = torch.zeros(1, 1, 4, dtype=torch.long)
        self.outputs, self.snapshots, self.events, self.applied = [], [], [], []
        original_apply = self.fly.manual.apply

        def apply(plan, layer, h):
            out = original_apply(plan, layer, h)
            self.applied[-1][layer] = (h.clone(), out.clone())
            return out
        self.fly.manual.apply = apply

    def step(self, frames=1, step_fn=None):
        for _ in range(frames):
            t = self.fly.frame
            self.applied.append({})
            source = torch.full((1, 1, DIMS), 0.01 * t)
            step = step_fn or self.fly.temporal_step
            h, self.self_kv, self.cross_kv = step(self.prev, self.self_kv, self.cross_kv, source)
            # Deterministic stand-in for token sampling: the next input depends on the output.
            self.prev = (h[0, 0, :4].abs() * 1000).long().remainder(32).view(1, 1, 4)
            self.outputs.append(h.clone())
            self.snapshots.append(self.fly.probe.collect())
            self.events.extend(self.fly.manual.resolve())
            self.clock.now += 0.04
        return self
