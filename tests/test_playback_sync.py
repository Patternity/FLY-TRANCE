"""Display aligned with the audible sound, on a fake audio clock."""
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication

from fly_trance.playback_sync import SAMPLE_RATE, PlaybackAligner


class Clock:
    def __init__(self):
        self.sample = 0

    def __call__(self):
        return self.sample


def make(clock, **kwargs):
    QCoreApplication.instance() or QCoreApplication([])
    shown = []
    aligner = PlaybackAligner(clock, lambda kind, payload, lag: shown.append((kind, payload, lag)), **kwargs)
    return aligner, shown


def test_states_are_shown_when_their_sound_starts_newest_first():
    clock = Clock()
    aligner, shown = make(clock)
    for sample, kind, name in ((1000, "telemetry", "t1"), (2000, "telemetry", "t2"),
                               (2500, "snapshot", "s1"), (3000, "telemetry", "t3")):
        aligner.push(sample, kind, name)
    assert shown == [] and aligner.timer.isActive()
    assert abs(aligner.timer.interval() - 1000 * 1000 / SAMPLE_RATE) <= 2  # armed for the first sound
    clock.sample = 2100
    aligner.flush()
    assert shown == [("telemetry", "t2", 100)]  # t1 was already outdated: skipped, not shown late
    clock.sample = 3000
    aligner.flush()
    assert sorted(shown[1:]) == [("snapshot", "s1", 500), ("telemetry", "t3", 0)]
    assert aligner.items == [] and not aligner.timer.isActive()


def test_out_of_order_arrivals_are_sorted_by_sound():
    clock = Clock()
    aligner, shown = make(clock)
    aligner.push(3000, "telemetry", "late")
    aligner.push(1500, "snapshot", "early")
    clock.sample = 2000
    aligner.flush()
    assert shown == [("snapshot", "early", 500)]


def test_without_audio_or_with_sync_off_states_show_at_generation_time():
    aligner, shown = make(lambda: None)
    aligner.push(5000, "telemetry", "x")
    assert shown == [("telemetry", "x", None)]
    clock = Clock()
    aligner, shown = make(clock)
    aligner.push(5000, "telemetry", "a")
    aligner.push(6000, "telemetry", "b")
    aligner.set_enabled(False)  # turning sync off shows the newest pending state at once
    assert shown == [("telemetry", "b", None)] and aligner.items == []
    aligner.push(9000, "snapshot", "c")
    assert shown[-1] == ("snapshot", "c", None)


def test_buffer_is_bounded_and_stop_drops_unheard_states():
    clock = Clock()
    aligner, shown = make(clock, capacity=4)
    for i in range(10):
        aligner.push(1000 * (i + 1), "telemetry", i)
    assert [item[3] for item in aligner.items] == [6, 7, 8, 9]
    clock.sample = None
    aligner.flush()  # the device is gone: queued sound will never be heard
    assert aligner.items == [] and shown == []


def test_a_stalled_device_keeps_states_waiting():
    clock = Clock()
    aligner, shown = make(clock)
    aligner.push(4800, "telemetry", "x")
    for _ in range(3):  # underrun: the audible position does not advance
        aligner.flush()
    assert shown == [] and aligner.timer.isActive()
    clock.sample = 4800
    aligner.flush()
    assert shown == [("telemetry", "x", 0)]
