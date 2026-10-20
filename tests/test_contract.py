from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fly_trance.audio import AudioOutput
from fly_trance.coupling import FlyConfig, FlyCoupling
from fly_trance.graph import load_graph


def coupling():
    graph, nodes, _ = load_graph()
    decoder = SimpleNamespace(cfg=SimpleNamespace(temporal=SimpleNamespace(model_dims=16)),
                              temporal_body=SimpleNamespace(layers=list(range(12))))
    return FlyCoupling(decoder, graph, nodes)


def test_authentic_graph_and_direction():
    graph, nodes, manifest = load_graph()
    assert graph.shape == (512, 512) and graph.nnz == 16731
    assert graph.sum() == 310999
    assert len({n['root_id'] for n in nodes}) == 512
    assert manifest['node_sha256'].startswith('62f91abd')


def test_fly_off_updates_brain_but_injection_is_identity():
    fly = coupling()
    h = torch.linspace(-1, 1, 16).reshape(1, 1, 16)
    fly._capture(h)
    before = fly.brain.state.copy()
    assert np.linalg.norm(before) > 0
    out, rms = fly._inject(h)
    assert out is h and rms == 0
    fly.configure(replace(fly.config, enabled=True))
    on, rms = fly._inject(h)
    assert rms > 0 and not torch.equal(on, h)
    np.testing.assert_array_equal(fly.brain.state, before)
    fly.configure(replace(fly.config, enabled=False))
    assert fly._inject(h)[0] is h
    np.testing.assert_array_equal(fly.brain.state, before)
    fly._capture(h)
    assert not np.array_equal(fly.brain.state, before)


def test_zero_strength_and_configuration_validation():
    fly = coupling()
    fly.configure(replace(fly.config, enabled=True, strength=0))
    h = torch.ones(1, 1, 16)
    assert fly._inject(h)[0] is h
    for config in (FlyConfig(strength=float('nan')), FlyConfig(leak=0), FlyConfig(inject=12)):
        with pytest.raises(ValueError):
            fly.configure(config)


def test_audio_queue_is_bounded_and_callback_handles_partial_chunks():
    sink = AudioOutput(sample_rate=100, capacity_seconds=0.1)
    sink.volume = 0.5
    chunk = np.ones((6, 2), dtype=np.float32)
    assert sink.push(chunk) and not sink.push(chunk)
    out = np.empty((4, 2), dtype=np.float32)
    sink._callback(out, 4, None, None)
    np.testing.assert_array_equal(out, 0.5)
    assert sink.queued == 2 and sink.played == 4
    sink._callback(out, 4, None, None)
    np.testing.assert_array_equal(out[:2], 0.5)
    np.testing.assert_array_equal(out[2:], 0)
    assert sink.queued == 0 and sink.underruns == 1
