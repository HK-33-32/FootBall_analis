from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "read_scoreboard", Path(__file__).resolve().parents[2] / "scripts" / "read_scoreboard.py"
)
read_scoreboard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(read_scoreboard)

_worker_spec = importlib.util.spec_from_file_location(
    "scoreboard_worker",
    Path(__file__).resolve().parents[2] / "scripts" / "_vlm_scoreboard_worker.py",
)
worker = importlib.util.module_from_spec(_worker_spec)
_worker_spec.loader.exec_module(worker)


def _reading(frame, home, away):
    return {"frame": frame, "home": home, "away": away}


def test_a_score_that_holds_for_two_frames_is_accepted():
    raw = [_reading(1, 0, 0), _reading(50, 1, 0), _reading(100, 1, 0)]
    timeline = read_scoreboard.monotonic(raw, "left")
    # dated to the frame the goal first showed, not to the one confirming it
    assert timeline[-1] == {"frame": 50, "left": 1, "right": 0}


def test_a_single_misread_frame_never_becomes_a_goal():
    raw = [_reading(1, 0, 0), _reading(50, 3, 0), _reading(100, 0, 0), _reading(150, 0, 0)]
    timeline = read_scoreboard.monotonic(raw, "left")
    assert all(entry["left"] == 0 for entry in timeline)


def test_a_score_is_never_allowed_to_fall():
    raw = [_reading(1, 1, 0), _reading(10, 1, 0), _reading(50, 0, 0), _reading(90, 1, 0)]
    timeline = read_scoreboard.monotonic(raw, "left")
    assert [entry["left"] for entry in timeline] == [1, 1]


def test_unreadable_frames_are_skipped():
    raw = [_reading(1, 0, 0), _reading(20, None, None), _reading(40, 0, 0)]
    assert [entry["frame"] for entry in read_scoreboard.monotonic(raw, "left")] == [1, 40]


def test_the_home_side_decides_which_pitch_side_the_numbers_land_on():
    raw = [_reading(1, 2, 1), _reading(10, 2, 1)]
    assert read_scoreboard.monotonic(raw, "right")[0] == {"frame": 1, "right": 2, "left": 1}
    assert read_scoreboard.monotonic(raw, "left")[0] == {"frame": 1, "left": 2, "right": 1}


def test_the_worker_reads_a_score_and_refuses_nonsense():
    assert worker.parse("2-1") == (2, 1)
    assert worker.parse("The score is 0 - 0.") == (0, 0)
    assert worker.parse("none") is None
    assert worker.parse("") is None
    assert worker.parse("45:12") is None  # the clock, not the score
