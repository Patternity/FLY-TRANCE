"""Manual pulses, holds, clear and block output bypass on the functional temporal path."""
import math
import os

import numpy as np
import pytest
import torch

from fly_trance.activations import BASE, CX512, MANUAL, RESULT
from fly_trance.manual import HOLD_TIMEOUT, MAX_PENDING_PULSES, MAX_STRENGTH, ManualMailbox
from toy_model import DIMS, LAYERS, ToyStream

GROUP_SIZE = 3  # grouping(130)


def manual_delta(stream, frame, layer):
    record = stream.applied[frame].get(layer)
    return None if record is None else (record[1] - record[0])


def frames_with_manual(stream, layer):
    return [t for t in range(len(stream.applied))
            if manual_delta(stream, t, layer) is not None and bool(manual_delta(stream, t, layer).abs().sum() > 0)]


def test_pulse_applies_on_exactly_one_frame_and_is_confirmed():
    stream = ToyStream(enabled=False).step(3)
    pulse = stream.mailbox.pulse(5, 7, 0.8)
    stream.step(5)
    assert frames_with_manual(stream, 5) == [3]
    assert [s.frame for s in stream.snapshots if np.any(s.stats[MANUAL] != 0)] == [3]
    assert stream.snapshots[3].manual_ops == ((5, 21, 24, 0.8, "pulse", pulse),)
    (event,) = stream.events
    assert event["type"] == "manual_pulse" and event["id"] == pulse and event["effective_frame"] == 3
    assert event["coordinates"] == [21, 24] and event["layer"] == 5


def test_addition_touches_only_the_selected_coordinates_with_the_documented_scale():
    stream = ToyStream(enabled=True).step(2)
    stream.mailbox.pulse(8, 10, -1.5)  # the CX512 injection layer: the reference includes CX512
    stream.step()
    before, after = stream.applied[2][8]
    delta = (after - before).reshape(-1)
    ref = before.float().square().mean().sqrt()
    outside = torch.ones(DIMS, dtype=torch.bool)
    outside[30:33] = False
    assert torch.equal(after.reshape(-1)[outside], before.reshape(-1)[outside])
    torch.testing.assert_close(delta[30:33], torch.full((3,), -1.5) * ref)
    event = stream.events[0]
    assert math.isclose(event["reference_rms"], float(ref), rel_tol=1e-6)
    assert math.isclose(event["per_coordinate"], -1.5 * float(ref), rel_tol=1e-6)
    s = stream.snapshots[2]
    assert s.stats[CX512, 8, 0] > 0 and s.stats[MANUAL, 8, 0] > 0
    np.testing.assert_allclose(s.manual_ref[8], float(ref), rtol=1e-6)


def test_hold_applies_every_frame_until_release_with_latest_strength():
    stream = ToyStream(enabled=False).step(2)
    hold = stream.mailbox.start_hold(4, 0, 0.5)
    stream.step(2)
    for strength in (0.6, 0.7, 0.9):  # several drag updates between frames: latest value wins
        stream.mailbox.update_hold(hold, strength)
    stream.step(2)
    stream.mailbox.release(hold)
    stream.step(3)
    assert frames_with_manual(stream, 4) == [2, 3, 4, 5]
    ref = stream.applied[4][4][0].float().square().mean().sqrt()
    torch.testing.assert_close(manual_delta(stream, 4, 4).reshape(-1)[:3], torch.full((3,), 0.9) * ref)
    kinds = [(e["type"], e["effective_frame"]) for e in stream.events]
    assert kinds[0] == ("manual_hold_start", 2) and kinds[-1] == ("manual_hold_end", 6)
    end = stream.events[-1]
    assert end["reason"] == "release" and end["frames_applied"] == 4 and end["final_strength"] == 0.9


def test_click_shorter_than_a_frame_still_applies_once():
    stream = ToyStream(enabled=False).step(1)
    hold = stream.mailbox.start_hold(2, 1, 1.0)
    stream.mailbox.release(hold)
    stream.step(3)
    assert frames_with_manual(stream, 2) == [1]


def test_stale_hold_is_released_by_the_watchdog():
    stream = ToyStream(enabled=False)
    hold = stream.mailbox.start_hold(3, 3, 0.4)
    stream.step(10)  # 0.4 s, still fresh
    stream.mailbox.update_hold(hold)
    stream.step(40)  # 1.6 s without a refresh
    applied = frames_with_manual(stream, 3)
    assert applied[0] == 0 and len(applied) < 50 and applied == list(range(applied[-1] + 1))
    end = [e for e in stream.events if e["type"] == "manual_hold_end"][0]
    assert end["reason"] == "timeout" and end["effective_frame"] == applied[-1] + 1
    # Released at the first frame boundary after HOLD_TIMEOUT since the last refresh (t = 0.4 s).
    assert HOLD_TIMEOUT - 1e-6 <= (end["effective_frame"] - 10) * 0.04 < HOLD_TIMEOUT + 0.04


def test_clear_cancels_pending_pulses_holds_and_bypass_without_touching_the_stream():
    stream = ToyStream(enabled=True).step(3)
    config, version = stream.fly.config, stream.fly.version
    stream.mailbox.set_bypass(6, True)
    stream.mailbox.start_hold(1, 1, 1.0)
    stream.step(2)
    brain = stream.fly.brain.state.copy()
    kv_length = stream.self_kv[0][0].shape[1]
    stream.mailbox.pulse(2, 2, 1.0)
    stream.mailbox.pulse(3, 3, 1.0)
    stream.mailbox.clear()
    stream.step(3)
    assert all(not stream.applied[t] or all(torch.equal(a, b) for a, b in stream.applied[t].values())
               for t in range(5, 8))
    assert all(s.bypass == () and np.all(s.stats[MANUAL] == 0) for s in stream.snapshots[5:])
    clear = [e for e in stream.events if e["type"] == "manual_clear"]
    assert clear and clear[0]["dropped_pulses"] == 2 and clear[0]["bypass_cleared"] == [6]
    assert clear[0]["released_hold"] is not None and clear[0]["effective_frame"] == 5
    # Fly settings, CX512, frame counter and caches continue; nothing was reset.
    assert stream.fly.config == config and stream.fly.version == version
    assert not np.array_equal(stream.fly.brain.state, brain) and np.linalg.norm(stream.fly.brain.state) > 0
    assert stream.fly.frame == 8 and stream.self_kv[0][0].shape[1] == kv_length + 3


def test_bypass_forwards_the_block_input_and_keeps_caches_advancing():
    plain = ToyStream(enabled=True).step(12)
    stream = ToyStream(enabled=True).step(3)
    stream.mailbox.set_bypass(4, True)
    stream.step(3)
    stream.mailbox.set_bypass(4, False)
    stream.step(6)
    for t in range(3, 6):
        # Under bypass, block 4 forwards its input: block 3's result, before any external addition.
        assert torch.equal(stream.applied[t][4][0], stream.applied[t][3][1])
        assert stream.snapshots[t].bypass == (4,)
        np.testing.assert_array_equal(stream.snapshots[t].mean[BASE, 4], stream.snapshots[t].mean[RESULT, 3])
    # The block still ran: at the first bypassed frame its input equals the plain run, and so do its caches.
    assert torch.equal(stream.self_kv[4][0][:, :4], plain.self_kv[4][0][:, :4])
    assert all(kv[0].shape[1] == 12 for kv in stream.self_kv)  # every block advanced every frame
    assert stream.fly.frame == 12 and not torch.equal(stream.outputs[3], plain.outputs[3])
    toggles = [(e["layer"], e["enabled"], e["effective_frame"]) for e in stream.events if e["type"] == "bypass"]
    assert toggles == [(4, True, 3), (4, False, 6)]


def test_fly_and_manual_control_are_independent():
    off = ToyStream(enabled=False).step(1)
    off.mailbox.start_hold(8, 5, 1.0)
    off.step(2)
    s = off.snapshots[-1]
    assert np.all(s.stats[CX512] == 0) and s.stats[MANUAL, 8, 0] > 0  # Fly OFF keeps manual control
    on = ToyStream(enabled=True).step(3)
    assert all(np.all(s.stats[MANUAL] == 0) and s.stats[CX512, 8, 0] > 0 for s in on.snapshots)
    assert np.linalg.norm(off.fly.brain.state) > 0  # CX512 kept listening with Fly OFF


def test_mailbox_bounds_and_validation():
    mailbox = ManualMailbox()
    with pytest.raises(RuntimeError):
        mailbox.pulse(0, 0, 1.0)
    mailbox.configure(LAYERS, DIMS)
    ids = [mailbox.pulse(0, 0, 1.0) for _ in range(MAX_PENDING_PULSES + 3)]
    assert None not in ids[:MAX_PENDING_PULSES] and ids[MAX_PENDING_PULSES:] == [None] * 3
    for bad in (float("nan"), float("inf"), MAX_STRENGTH + 0.01):
        with pytest.raises(ValueError):
            mailbox.pulse(0, 0, bad)
    for layer, group in ((LAYERS, 0), (0, 44), (-1, 0)):
        with pytest.raises(ValueError):
            mailbox.start_hold(layer, group, 1.0)
    hold = mailbox.start_hold(0, 0, 1.0)
    assert not mailbox.update_hold(hold + 100, 2.0)
    mailbox.release(hold + 100)
    assert mailbox.pending() == (MAX_PENDING_PULSES, True)
    mailbox.reset_session()
    assert mailbox.pending() == (0, False)


def test_stop_closes_an_active_hold_in_the_journal():
    stream = ToyStream(enabled=False)
    stream.mailbox.start_hold(0, 0, 1.0)
    stream.step(3)
    (end,) = stream.fly.manual.end_session(stream.fly.frame, "stop")
    assert end["type"] == "manual_hold_end" and end["reason"] == "stop" and end["frames_applied"] == 3
    stream.mailbox.reset_session()
    stream.step(2)
    assert frames_with_manual(stream, 0) == [0, 1, 2]


def test_journal_matches_the_applied_operations():
    stream = ToyStream(enabled=True)
    stream.step(1)
    for frame in range(1, 12):
        if frame in (2, 5, 9):  # queued before that frame is computed
            stream.mailbox.pulse(frame % LAYERS, frame, 0.3 * frame / 3)
        stream.step()
    pulses = [e for e in stream.events if e["type"] == "manual_pulse"]
    assert [e["effective_frame"] for e in pulses] == [2, 5, 9]
    for event in pulses:
        before, after = stream.applied[event["effective_frame"]][event["layer"]]
        a, b = event["coordinates"]
        ref = float(before.float().square().mean().sqrt())
        assert math.isclose(event["reference_rms"], ref, rel_tol=1e-6)
        np.testing.assert_allclose((after - before).reshape(-1)[a:b].numpy(), event["per_coordinate"], rtol=1e-5)
        assert stream.snapshots[event["effective_frame"]].manual_ops[0][:3] == (event["layer"], a, b)


def test_panel_gestures_send_bounded_commands_and_release_on_focus_loss():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtGui import QFocusEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from fly_trance.activity_view import TransformerActivityPanel
    app = QApplication.instance() or QApplication([])
    stream = ToyStream(enabled=True).step(2)
    panel = TransformerActivityPanel()
    panel.attach_manual(stream.mailbox)
    panel.set_model(LAYERS, DIMS)
    panel.resize(360, 900)
    s = stream.snapshots[-1]
    s.timing = {"generated_seconds": 1.0, "played_seconds": None}
    panel.update_snapshot(s)
    panel.show()
    app.processEvents()
    m = panel.map
    panel.manual_more.setChecked(True)
    app.processEvents()
    cell = m._cell_rectf(5, 10).center().toPoint()
    # Safe viewing by default: a click selects a column and sends nothing.
    QTest.mouseClick(m, Qt.LeftButton, Qt.NoModifier, cell)
    assert stream.mailbox.pending() == (0, False) and panel.selected == 5
    assert "Enable mouse intervention" in panel.manual_info.text()  # says what to switch on
    panel.manual_mode.setChecked(True)
    assert m.cursor().shape() == Qt.CrossCursor
    panel.manual_kind.setCurrentIndex(1)  # Hold
    QTest.mousePress(m, Qt.LeftButton, Qt.NoModifier, cell)
    hold = panel._gesture["id"]
    assert stream.mailbox._hold.layer == 5 and stream.mailbox._hold.group == 10
    QTest.mouseMove(m, cell - QPoint(0, 100))  # 100 px up: +1.0 on top of the starting +0.5
    assert math.isclose(stream.mailbox._hold.strength, 1.5)
    QTest.mouseRelease(m, Qt.LeftButton, Qt.NoModifier, cell - QPoint(0, 100))
    assert stream.mailbox._hold is None or stream.mailbox._hold.release_after_apply
    stream.mailbox.take()
    # Focus loss and hiding release a pressed hold even without a mouse release.
    QTest.mousePress(m, Qt.LeftButton, Qt.NoModifier, cell)
    assert stream.mailbox._hold is not None and stream.mailbox._hold.id != hold
    QApplication.sendEvent(m, QFocusEvent(QEvent.FocusOut, Qt.ActiveWindowFocusReason))
    assert stream.mailbox._hold is None or stream.mailbox._hold.release_after_apply
    assert stream.mailbox.release_reason(max(stream.mailbox._reasons)) == "focus lost"
    stream.mailbox.take()
    QTest.mousePress(m, Qt.RightButton, Qt.NoModifier, cell)  # right button: opposite sign
    assert stream.mailbox._hold.strength == -0.5
    panel.hide()
    assert stream.mailbox._hold is None or stream.mailbox._hold.release_after_apply
    panel.show()
    QTest.mouseRelease(m, Qt.RightButton, Qt.NoModifier, cell)  # the late release is ignored
    assert stream.mailbox._hold is None or stream.mailbox._hold.release_after_apply
    stream.mailbox.take()
    # Pulse mode: one pulse per click, sent on release.
    panel.manual_kind.setCurrentIndex(0)
    QTest.mouseClick(m, Qt.LeftButton, Qt.NoModifier, cell)
    assert stream.mailbox.pending() == (1, False)
    panel._bypass_boxes[3].setChecked(True)
    assert stream.mailbox.bypass() == frozenset({3})
    panel.clear_interventions()
    assert stream.mailbox.pending() == (0, False) and stream.mailbox.bypass() == frozenset()
    assert not panel._bypass_boxes[3].isChecked()
    panel.manual_mode.setChecked(False)
    QTest.mouseClick(m, Qt.LeftButton, Qt.NoModifier, cell)
    assert stream.mailbox.pending() == (0, False)
    panel.on_manual_event({"type": "bypass", "effective_frame": 3425, "layer": 0, "enabled": False})
    panel.new_session()
    assert "3425" not in panel.manual_confirm.text()
    panel.close()
