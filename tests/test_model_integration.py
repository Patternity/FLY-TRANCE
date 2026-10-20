"""Opt-in integration test, uses existing local weights without a download."""
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from fly_trance.activations import BASE, CX512, MANUAL, RESULT
from fly_trance.engine import MusicEngine
from fly_trance.manual import ManualMailbox
from toy_model import original_temporal_step


@pytest.mark.skipif(not os.environ.get('FLY_MODEL_DIR'), reason='Set FLY_MODEL_DIR to local MRT2 Small')
def test_actual_model_off_identity_toggle_and_bounded_state():
    engine = MusicEngine(os.environ['FLY_MODEL_DIR'], device='cpu')
    decoder = engine.model.depthformer.decoder
    # Compare the actual unmodified step to the replacement on identical tokens/KV.
    state = decoder.init_streaming_f(1, engine.model._dev, engine.model._dt)
    with torch.no_grad():
        expected = decoder.temporal_step_fn(state['prev'], state['self'], state['cross'], engine.source)
        actual = engine.coupling.temporal_step(state['prev'], state['self'], state['cross'], engine.source)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    for kind in (1, 2):
        for a, b in zip(actual[kind], expected[kind]):
            for aa, bb in zip(a, b):
                torch.testing.assert_close(aa, bb, rtol=0, atol=0)
    engine.reset(42)
    shapes = []
    for i in range(56):
        if i in (2, 8):
            brain = engine.coupling.brain.state.copy()
            cache = engine.dstate
            codec = engine.decode_state
            rng = engine.generator.get_state().clone()
            engine.configure(replace(engine.coupling.config, enabled=i == 2))
            np.testing.assert_array_equal(brain, engine.coupling.brain.state)
            assert engine.dstate is cache and engine.decode_state is codec
            assert torch.equal(rng, engine.generator.get_state())
        audio, snapshot = engine.frame()
        assert np.isfinite(audio).all()
        assert (snapshot['delta_rms'] > 0) == (2 <= i < 8)
        shapes.append(engine.dstate['self'][0][0].shape[1])
    assert shapes[-1] == decoder.cfg.temporal_max_past + 1
    assert engine.coupling.frame == 56 and np.linalg.norm(snapshot['state']) > 0


needs_model = pytest.mark.skipif(not os.environ.get('FLY_MODEL_DIR'), reason='Set FLY_MODEL_DIR to local MRT2 Small')


@pytest.fixture(scope='module')
def engine():
    return MusicEngine(os.environ['FLY_MODEL_DIR'], device=os.environ.get('FLY_TEST_DEVICE', 'cpu'))


@needs_model
def test_actual_model_activation_observation_is_read_only(engine):
    decoder = engine.model.depthformer.decoder
    probe = engine.coupling.probe
    # Read from the loaded model, not assumed.
    assert probe.layers == len(decoder.temporal_body.layers) == 12
    assert probe.dims == decoder.cfg.temporal.model_dims
    config = replace(engine.coupling.config, enabled=True, strength=0.3, capture=9, inject=7)
    frames = 40

    def session(observe_at):
        engine.reset(42)
        engine.configure(config)
        engine.set_observe(False)
        probe.interval = 0  # every frame: the strictest read-only check
        tokens, audio, snapshots = [], [], []
        for i in range(frames):
            if i in observe_at:
                cache, codec = engine.dstate, engine.decode_state
                rng = engine.generator.get_state().clone()
                brain, frame = engine.coupling.brain.state.copy(), engine.coupling.frame
                engine.set_observe(observe_at[i])
                assert engine.dstate is cache and engine.decode_state is codec
                assert torch.equal(rng, engine.generator.get_state())
                np.testing.assert_array_equal(brain, engine.coupling.brain.state)
                assert engine.coupling.frame == frame
            if i == frames // 2:
                engine.configure(replace(config, enabled=False))
            wav, _ = engine.frame()
            tokens.append(engine.dstate['prev'].cpu().clone())
            audio.append(wav)
            snapshots.append(engine.take_activations())
        return torch.cat(tokens, 1), np.concatenate(audio), snapshots, engine.coupling.brain.state.copy()

    ref_tokens, ref_audio, ref_snaps, ref_brain = session({})
    assert all(s is None for s in ref_snaps)
    for observe_at in ({0: True}, {5: True, 15: False, 25: True}):
        tokens, audio, snaps, brain = session(observe_at)
        assert torch.equal(tokens, ref_tokens)
        np.testing.assert_array_equal(brain, ref_brain)
        # Same tokens through the same codec path; a tolerance is allowed only for audio.
        np.testing.assert_allclose(audio, ref_audio, rtol=0, atol=1e-6)
    observed = [(i, s) for i, s in enumerate(snaps) if s is not None]
    assert [i for i, _ in observed] == list(range(5, 15)) + list(range(25, frames))
    for i, s in observed:
        assert s.frame == i and s.layers == 12 and s.mean.shape == (4, 12, 64) and s.group_size == s.dims // 64
        for array in (s.mean, s.rms, s.stats, s.manual_ref):
            assert np.isfinite(array).all()
        assert s.config['capture'] == 9 and s.config['inject'] == 7
        assert np.all(s.stats[MANUAL] == 0) and np.all(np.delete(s.stats[CX512], 7, axis=0) == 0)
        if s.config['enabled']:
            assert s.stats[CX512, 7, 0] > 0 and 0 < s.delta_ratio(CX512, 7) < 1
        else:
            assert np.all(s.stats[CX512] == 0) and np.all(s.mean[CX512] == 0) and np.all(s.rms[CX512] == 0)
            np.testing.assert_array_equal(s.stats[BASE], s.stats[RESULT])
    assert any(s.config['enabled'] for _, s in observed) and any(not s.config['enabled'] for _, s in observed)

    # Independent reference: FFN modules are called as modules on this path, so hooks see
    # the same post-FFN residual for every layer of the observed frame.
    engine.reset(42)
    engine.configure(config)
    engine.set_observe(True)
    seen = []
    handles = [block['ffn'].register_forward_hook(lambda m, i, o: seen.append(o.detach().float().cpu()))
               for block in decoder.temporal_body.layers]
    try:
        for i in range(3):
            seen.clear()
            engine.frame()
            s = engine.take_activations()
            vectors = torch.stack([o.reshape(-1) for o in seen]).numpy()
            assert s.frame == i == engine.coupling.frame - 1 and len(seen) == 12
            np.testing.assert_allclose(s.stats[BASE, :, 0], np.sqrt((vectors ** 2).mean(-1)), rtol=1e-4)
            np.testing.assert_allclose(s.stats[BASE, :, 1], vectors.min(-1), rtol=1e-6)
            np.testing.assert_allclose(s.stats[BASE, :, 2], vectors.max(-1), rtol=1e-6)
            np.testing.assert_allclose(s.mean[BASE], vectors.reshape(12, 64, -1).mean(-1), rtol=1e-4, atol=1e-5)
            # The nominal and measured addition agree up to the model dtype's rounding.
            nominal = engine.coupling.last['delta_rms']
            assert abs(s.stats[CX512, 7, 0] - nominal) <= 0.25 * nominal
    finally:
        for handle in handles:
            handle.remove()


@needs_model
def test_actual_model_manual_control(engine):
    config = replace(engine.coupling.config, enabled=True, strength=0.3, capture=9, inject=7)
    probe = engine.coupling.probe
    frames = 30

    def session(script=None, original=False):
        mailbox = ManualMailbox()
        engine.coupling.manual.attach(mailbox)
        engine.reset(42)
        engine.configure(config)
        engine.set_observe(not original)
        probe.interval = 0
        if original:
            engine.coupling.temporal_step = lambda *a: original_temporal_step(engine.coupling, *a)
        tokens, audio, snaps, events = [], [], [], []
        try:
            for i in range(frames):
                if script:
                    script(i, mailbox)
                wav, _ = engine.frame()
                tokens.append(engine.dstate['prev'].cpu().clone())
                audio.append(wav)
                snaps.append(engine.take_activations())
                events.extend(engine.take_events())
                assert all(k.shape[1] == min(i + 1, decoder_keep) for k, _ in engine.dstate['self'])
        finally:
            engine.coupling.__dict__.pop('temporal_step', None)
        return torch.cat(tokens, 1), np.concatenate(audio), snaps, events

    decoder_keep = engine.model.depthformer.decoder.cfg.temporal_max_past + 1
    # With nothing manual active, generation is token-identical to the released v0.1.0 step.
    idle_tokens, idle_audio, _, idle_events = session()
    old_tokens, old_audio, _, _ = session(original=True)
    assert torch.equal(idle_tokens, old_tokens) and idle_events == []
    np.testing.assert_allclose(idle_audio, old_audio, rtol=0, atol=1e-6)

    def script(i, mailbox):
        if i == 5:
            mailbox.pulse(3, 10, 1.0)
        elif i == 10:
            mailbox.set_bypass(6, True)
        elif i == 20:
            mailbox.set_bypass(6, False)
        elif i == 22:
            script.hold = mailbox.start_hold(9, 2, -0.8)
        elif i == 26:
            mailbox.release(script.hold)
        elif i == 28:
            mailbox.pulse(1, 1, 1.0)
            mailbox.clear()

    tokens, audio, snaps, events = session(script)
    assert np.isfinite(audio).all()
    assert torch.equal(tokens[:, :5], idle_tokens[:, :5])  # nothing changed before the first pulse
    manual_frames = [s.frame for s in snaps if np.any(s.stats[MANUAL] != 0)]
    assert manual_frames == [5, 22, 23, 24, 25]  # one pulse frame, four hold frames, nothing after clear
    for s in snaps:
        layers = sorted({op[0] for op in s.manual_ops})
        for layer in range(12):
            if layer not in layers:
                assert np.all(s.stats[MANUAL, layer] == 0)
        for layer, start, stop, *_ in s.manual_ops:
            # Only the targeted coordinate group moved; every other group is exactly unchanged.
            assert np.flatnonzero(s.rms[MANUAL, layer]).tolist() == [start // s.group_size]
            assert s.manual_ref[layer] > 0
        # Fly keeps injecting independently of manual control. Read (10) follows inject (8), so on
        # frame 0 the CX512 state is still zero and nothing is added yet.
        assert (s.stats[CX512, 7, 0] > 0) == (s.frame > 0)
        assert np.isfinite(s.mean).all() and np.isfinite(s.stats).all()
        if 10 <= s.frame < 20:
            assert s.bypass == (6,)
            np.testing.assert_array_equal(s.mean[BASE, 6], s.mean[RESULT, 5])
            np.testing.assert_array_equal(s.stats[BASE, 6], s.stats[RESULT, 5])
        else:
            assert s.bypass == ()
    journal = [(e['type'], e['effective_frame']) for e in events]
    assert journal == [('manual_pulse', 5), ('bypass', 10), ('bypass', 20), ('manual_hold_start', 22),
                       ('manual_hold_end', 26), ('manual_clear', 28)]
    assert events[-1]['dropped_pulses'] == 1 and events[0]['reference_rms'] > 0
    assert engine.coupling.frame == frames
