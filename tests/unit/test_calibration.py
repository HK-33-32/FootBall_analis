from __future__ import annotations

from football_intelligence.calibration import (
    CalibrationConfig,
    drop_uncalibrated,
    frame_validity,
)


def _person(frame, image_x, pitch_x, pitch_y):
    return {
        "frame": frame,
        "track_id": int(image_x),
        "bbox_image": {"x": float(image_x), "y": 500.0, "w": 40.0, "h": 90.0},
        "bbox_pitch": {"x_bottom_middle": float(pitch_x), "y_bottom_middle": float(pitch_y)},
        "attributes": {"role": "player", "team": "left", "jersey": None},
    }


def _spread(frame, mirrored=False, scale=1.0, offset=0.0):
    """Ten players strung across the image, mapped onto the pitch."""
    rows = []
    for index in range(10):
        image_x = 200 + index * 140
        pitch_x = (-40 + index * 9) * scale + offset
        rows.append(_person(frame, image_x, -pitch_x if mirrored else pitch_x, index - 5))
    return rows


def test_a_well_calibrated_clip_keeps_every_frame():
    predictions = [row for frame in (1, 2, 3) for row in _spread(frame)]
    report = frame_validity(predictions, CalibrationConfig())
    assert report["frames_rejected"] == 0
    assert report["frames_examined"] == 3


def test_a_mirrored_frame_is_caught_against_the_rest_of_the_clip():
    good = [row for frame in range(1, 9) for row in _spread(frame)]
    bad = _spread(9, mirrored=True)
    report = frame_validity(good + bad, CalibrationConfig())
    assert list(report["rejected"]) == [9]
    assert report["rejected"][9] == "pitch mapping is mirrored"


def test_the_clip_decides_its_own_orientation():
    """A clip that is mirrored throughout is self-consistent, not broken."""
    predictions = [row for frame in range(1, 9) for row in _spread(frame, mirrored=True)]
    assert frame_validity(predictions, CalibrationConfig())["frames_rejected"] == 0


def test_a_frame_that_collapses_everyone_onto_one_spot_is_caught():
    good = [row for frame in range(1, 9) for row in _spread(frame)]
    collapsed = [_person(9, 200 + index * 140, 0.1 * index, 0.0) for index in range(10)]
    report = frame_validity(good + collapsed, CalibrationConfig())
    assert report["rejected"][9] == "everyone collapsed onto one spot"


def test_players_projected_off_the_pitch_condemn_the_frame():
    good = [row for frame in range(1, 9) for row in _spread(frame)]
    outside = [_person(9, 200 + index * 140, 90.0 + index, 45.0) for index in range(10)]
    report = frame_validity(good + outside, CalibrationConfig())
    assert report["rejected"][9] == "players projected off the pitch"


def test_frames_with_too_few_people_are_left_alone():
    """Two players cannot tell you whether a homography is sane."""
    good = [row for frame in range(1, 9) for row in _spread(frame)]
    sparse = [_person(9, 200, 40.0, 0.0), _person(9, 900, -40.0, 0.0)]
    report = frame_validity(good + sparse, CalibrationConfig())
    assert 9 not in report["rejected"]


def test_dropping_removes_rejected_frames_and_stray_points():
    good = [row for frame in range(1, 9) for row in _spread(frame)]
    bad = _spread(9, mirrored=True)
    stray = _person(3, 400, 300.0, 0.0)  # one impossible point on a good frame
    kept, report = drop_uncalibrated(good + bad + [stray], CalibrationConfig())
    assert report["frames_rejected"] == 1
    assert report["detections_off_pitch"] == 1
    assert all(int(detection["frame"]) != 9 for detection in kept)
    assert len(kept) == len(good)


def test_the_ball_never_votes_on_calibration():
    good = [row for frame in range(1, 9) for row in _spread(frame)]
    ball = {
        "frame": 3,
        "track_id": 99,
        "bbox_image": {"x": 10.0, "y": 10.0, "w": 6.0, "h": 6.0},
        "bbox_pitch": {"x_bottom_middle": 400.0, "y_bottom_middle": 400.0},
        "attributes": {"role": "ball", "team": None, "jersey": None},
    }
    report = frame_validity(good + [ball], CalibrationConfig())
    assert report["frames_rejected"] == 0
