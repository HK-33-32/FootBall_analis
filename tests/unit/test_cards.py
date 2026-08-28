from __future__ import annotations

import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parents[2] / "scripts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


read_cards = _load("read_cards", _root / "read_cards.py")
worker = _load("card_worker", _root / "_vlm_card_worker.py")


def test_frames_are_asked_for_around_each_stoppage():
    report = {"stoppages": [{"frame": 100}, {"frame": 500}]}
    frames = read_cards.frames_to_ask(report, fps=25.0, spread_s=4.0, per_stoppage=3)
    assert frames[0] == 100
    assert 500 in frames
    assert len(frames) == 6
    assert frames == sorted(frames)


def test_a_report_without_stoppages_asks_nothing():
    assert read_cards.frames_to_ask({"stoppages": []}, 25.0, 4.0, 3) == []


def test_one_booking_seen_on_several_frames_is_one_card():
    readings = [
        {"frame": 100, "colour": "yellow"},
        {"frame": 125, "colour": "yellow"},
        {"frame": 150, "colour": "yellow"},
    ]
    assert len(read_cards.collapse(readings, fps=25.0)) == 1


def test_two_bookings_far_apart_stay_two_cards():
    readings = [{"frame": 100, "colour": "yellow"}, {"frame": 900, "colour": "yellow"}]
    assert len(read_cards.collapse(readings, fps=25.0)) == 2


def test_frames_with_no_card_are_dropped():
    readings = [{"frame": 100, "colour": None}, {"frame": 200, "colour": "red"}]
    cards = read_cards.collapse(readings, fps=25.0)
    assert [card["colour"] for card in cards] == ["red"]


def test_the_worker_reads_the_one_word_it_asked_for():
    assert worker.parse("yellow") == "yellow"
    assert worker.parse("Red card.") == "red"
    assert worker.parse("none") is None
    assert worker.parse("") is None
