"""Activation observation on the functional temporal path, without model weights."""
import os

import numpy as np
import pytest
import torch

from fly_trance.activations import BASE, CX512, MANUAL, RESULT, ActivationProbe, LatestSlot, grouping
from toy_model import DIMS, LAYERS, ToyStream, original_temporal_step


def run(observe, enabled=True, frames=10, toggle_at=None):
    stream = ToyStream(enabled=enabled, observe=observe)
    for t in range(frames):
        if t == toggle_at:
            stream.fly.probe.set_enabled(not stream.fly.probe.enabled)
        stream.step()
    return stream


def test_grouping_is_deterministic_and_compact():
    assert grouping(1024) == (16, 64)
    assert grouping(3072) == (48, 64)
    assert grouping(130) == (3, 44)
    assert grouping(16) == (1, 16)


def test_observation_does_not_change_model_values():
    off, on = run(observe=False), run(observe=True)
    for a, b in zip(off.outputs, on.outputs):
        assert torch.equal(a, b)
    np.testing.assert_array_equal(off.fly.brain.state, on.fly.brain.state)
    assert all(s is None for s in off.snapshots)
    assert all(s is not None for s in on.snapshots)


def test_snapshot_matches_one_frame_at_the_documented_position():
    # Reference: forward hooks on the toy FFN modules see the same post-FFN residual.
    stream = ToyStream(enabled=True, observe=True)
    hooks = []
    for block in stream.decoder.temporal_body.layers:
        block["ffn"].register_forward_hook(lambda m, i, o: hooks.append(o.detach().clone()))
    for t in range(4):
        hooks.clear()
        stream.step()
        s = stream.snapshots[-1]
        assert s.frame == t == stream.fly.frame - 1
        assert s.layers == LAYERS and s.dims == DIMS and s.mean.shape == (4, LAYERS, 44)
        assert len(hooks) == LAYERS
        vectors = torch.stack([o.reshape(-1) for o in hooks]).numpy()
        padded = np.pad(vectors, ((0, 0), (0, 44 * 3 - DIMS)))
        counts = np.array([3] * 43 + [1])
        np.testing.assert_allclose(s.mean[BASE], padded.reshape(LAYERS, 44, 3).sum(-1) / counts, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(s.rms[BASE], np.sqrt((padded ** 2).reshape(LAYERS, 44, 3).sum(-1) / counts),
                                   rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(s.stats[BASE, :, 0], np.sqrt((vectors ** 2).mean(-1)), rtol=1e-5)
        np.testing.assert_allclose(s.stats[BASE, :, 1], vectors.min(-1), rtol=1e-6)
        np.testing.assert_allclose(s.stats[BASE, :, 2], vectors.max(-1), rtol=1e-6)
        # CX512 adds only at the injection layer; the result there is base plus that delta.
        assert s.stats[CX512, 8, 0] > 0 and np.isclose(s.stats[CX512, 8, 0], stream.fly.last["delta_rms"], rtol=1e-4)
        assert np.all(np.delete(s.stats[CX512], 8, axis=0) == 0) and np.all(s.stats[MANUAL] == 0)
        np.testing.assert_allclose(s.mean[RESULT, 8], s.mean[BASE, 8] + s.mean[CX512, 8], atol=1e-5)
        for array in (s.mean, s.rms, s.stats, s.manual_ref):
            assert np.isfinite(array).all()
        assert s.config_version == stream.fly.version and s.config["inject"] == 8 and s.config["capture"] == 3


def test_fly_off_adds_exactly_zero_while_brain_keeps_listening():
    stream = run(observe=True, enabled=False)
    assert np.linalg.norm(stream.fly.brain.state) > 0
    for s in stream.snapshots:
        assert not s.config["enabled"]
        assert np.all(s.mean[CX512] == 0) and np.all(s.rms[CX512] == 0) and np.all(s.stats[CX512] == 0)
        np.testing.assert_array_equal(s.mean[BASE], s.mean[RESULT])


def test_toggling_observation_mid_stream_keeps_the_stream():
    ref, toggled = run(observe=False), run(observe=False, toggle_at=5)
    for a, b in zip(ref.outputs, toggled.outputs):
        assert torch.equal(a, b)
    np.testing.assert_array_equal(ref.fly.brain.state, toggled.fly.brain.state)
    assert [s is not None for s in toggled.snapshots] == [False] * 5 + [True] * 5
    assert toggled.fly.frame == 10 and toggled.snapshots[5].frame == 5


def test_new_features_off_match_the_released_implementation():
    # Observation on, manual control attached but idle: bit-identical to v0.1.0's temporal step.
    for enabled in (True, False):
        new = ToyStream(enabled=enabled, observe=True)
        old = ToyStream(enabled=enabled, observe=False, attach=False)
        new.step(10)
        old.step(10, step_fn=lambda *a: original_temporal_step(old.fly, *a))
        for a, b in zip(new.outputs, old.outputs):
            assert torch.equal(a, b)
        for (ka, va), (kb, vb) in zip(new.self_kv, old.self_kv):
            assert torch.equal(ka, kb) and torch.equal(va, vb)
        np.testing.assert_array_equal(new.fly.brain.state, old.fly.brain.state)
        assert new.events == []


def test_frame_selection_is_rate_limited_and_off_does_nothing():
    for frame_seconds in (0.04, 0.0436, 0.02):  # realtime, the measured RTF 1.09, and faster
        clock = [0.0]
        probe = ActivationProbe(LAYERS, DIMS, interval=0.2, clock=lambda: clock[0])
        assert probe.begin(0) is None
        probe.set_enabled(True)
        picked = []
        for frame in range(round(10 / frame_seconds)):  # ten seconds of wall time
            if probe.begin(frame) is not None:
                picked.append(frame)
            clock[0] += frame_seconds
        assert 49 <= len(picked) <= 51
        probe.set_enabled(False)
        assert probe.begin(10 ** 6) is None and probe.collect() is None


def test_latest_slot_is_bounded_and_wakes_once():
    slot = LatestSlot()
    assert slot.put(1) is True
    assert slot.put(2) is False and slot.put(3) is False
    assert slot.dropped == 2 and slot.take() == 3 and slot.take() is None
    assert slot.put(4) is True


def test_panel_renders_measured_snapshot_offscreen():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from fly_trance.activity_view import (NEGATIVE, POSITIVE, ZERO, TransformerActivityPanel, compose_image,
                                          diverging_rgb, energy_rgb)
    rgb, sat = diverging_rgb(np.array([-2.0, -1.0, 0.0, 1.0]), 1.0)
    np.testing.assert_array_equal(rgb[1], NEGATIVE.astype(np.uint8))
    np.testing.assert_array_equal(rgb[2], ZERO.astype(np.uint8))
    np.testing.assert_array_equal(rgb[3], POSITIVE.astype(np.uint8))
    assert sat.tolist() == [True, False, False, False]
    assert energy_rgb(np.array([0.5, 3.0]), 1.0)[1].tolist() == [False, True]
    assert compose_image(np.zeros((12, 64)), np.ones((12, 64)), (1.0, 1.0)).shape == (64, 120, 3)
    app = QApplication.instance() or QApplication([])
    stream = run(observe=True, frames=3)
    panel = TransformerActivityPanel()
    panel.resize(360, 700)
    s = stream.snapshots[-1]
    s.timing = {"generated_seconds": 1.0, "played_seconds": 0.2, "queue_seconds": 0.8}
    panel.update_snapshot(s)
    panel.more.setChecked(True)
    panel.select_layer(3)
    assert not panel.grab().isNull()
    assert "Layer 4" in panel.layer_info.text() and "read" in panel.layer_info.text()
    assert "layer 9" in panel.legend.text()
    app.processEvents()
