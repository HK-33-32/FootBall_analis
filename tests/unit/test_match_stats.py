from __future__ import annotations

from football_intelligence.match_stats import StatsConfig, match_statistics, timeline


def _detection(frame, track, x, y, role="player", team="left", jersey=None):
    return {
        "frame": frame,
        "track_id": track,
        "bbox_image": {"x": 10.0, "y": 10.0, "w": 20.0, "h": 60.0},
        "bbox_pitch": {"x_bottom_middle": x, "y_bottom_middle": y},
        "attributes": {"role": role, "team": team, "jersey": jersey},
    }


def _walker(track, frames, speed, y=0.0, **kwargs):
    return [
        _detection(frame, track, index * speed / 25.0, y, **kwargs)
        for index, frame in enumerate(frames)
    ]


def test_statistics_are_measured_only_over_frames_that_have_a_game_state():
    predictions = _walker(1, list(range(1, 26)), 4.0, jersey="9")
    report = match_statistics(predictions, StatsConfig())
    player = report["players"][0]
    assert player["jersey"] == "9"
    assert player["time_on_camera_s"] == 1.0
    assert report["coverage"] == 1.0
    # 4 m/s for one second is 3.84 m; smoothing trims the ends of a track this
    # short, and under-reporting is the right side to err on for distance
    assert 2.8 < player["distance_m"] < 4.0


def test_a_sprint_carries_the_timecode_it_came_from():
    slow = _walker(1, list(range(1, 26)), 1.0)
    fast = [_detection(frame, 1, 1.0 + (frame - 25) * 9.0 / 25.0, 0.0) for frame in range(26, 76)]
    report = match_statistics(slow + fast, StatsConfig())
    sprints = report["players"][0]["sprints"]
    assert sprints, "a two-second run at 9 m/s is a sprint"
    assert sprints[0]["start_timecode"].startswith("00:0")
    assert sprints[0]["duration_s"] >= 0.6
    assert sprints[0]["top_speed_kmh"] > 25


def test_visible_segments_split_where_the_camera_looked_away():
    frames = list(range(1, 21)) + list(range(200, 221))
    report = match_statistics(_walker(1, frames, 2.0), StatsConfig())
    segments = report["players"][0]["visible_segments"]
    assert len(segments) == 2
    assert segments[0]["end_frame"] == 20
    assert segments[1]["start_frame"] == 200


def test_possession_goes_to_the_closest_player_inside_the_radius():
    near = [_detection(frame, 1, 0.0, 0.0, team="left") for frame in range(1, 26)]
    far = [_detection(frame, 2, 30.0, 0.0, team="right") for frame in range(1, 26)]
    ball = [_detection(frame, 99, 0.5, 0.0, role="ball", team=None) for frame in range(1, 26)]
    report = match_statistics(near + far + ball, StatsConfig())
    by_id = {player["identity"]: player for player in report["players"]}
    assert by_id[1]["time_nearest_ball_s"] > 0.9
    assert by_id[2]["time_nearest_ball_s"] == 0.0
    assert report["teams"]["left"]["possession_share"] == 1.0


def test_a_ball_nobody_is_near_belongs_to_nobody():
    away = [_detection(frame, 1, 40.0, 20.0) for frame in range(1, 26)]
    ball = [_detection(frame, 99, 0.0, 0.0, role="ball", team=None) for frame in range(1, 26)]
    report = match_statistics(away + ball, StatsConfig())
    assert report["players"][0]["time_nearest_ball_s"] == 0.0


def test_referees_never_take_possession():
    referee = [_detection(frame, 1, 0.0, 0.0, role="referee", team=None) for frame in range(1, 26)]
    ball = [_detection(frame, 99, 0.2, 0.0, role="ball", team=None) for frame in range(1, 26)]
    report = match_statistics(referee + ball, StatsConfig())
    assert report["players"][0]["role"] == "referee"
    assert report["players"][0]["time_nearest_ball_s"] == 0.0


def test_thirds_are_named_from_the_player_own_direction():
    deep_left = [_detection(frame, 1, -40.0, 0.0, team="left") for frame in range(1, 26)]
    deep_right = [_detection(frame, 2, -40.0, 0.0, team="right") for frame in range(1, 26)]
    report = match_statistics(deep_left + deep_right, StatsConfig())
    by_id = {player["identity"]: player for player in report["players"]}
    assert by_id[1]["thirds"]["attacking"] == 1.0
    assert by_id[2]["thirds"]["defensive"] == 1.0


def test_timeline_is_keyed_by_frame_and_excludes_the_ball_from_people():
    predictions = _walker(1, [1, 2], 2.0) + [_detection(1, 99, 5.0, 5.0, role="ball", team=None)]
    result = timeline(predictions, [1], [(5.0, 5.0)])
    assert result["frames"] == [1, 2]
    assert [row[0] for row in result["people"]["1"]] == [1]
    assert result["ball"]["1"] == [5.0, 5.0]


def test_speed_zones_split_the_distance_by_how_it_was_covered():
    walk = [_detection(frame, 1, frame * 1.0 / 25.0, 0.0) for frame in range(1, 76)]
    sprint = [
        _detection(frame, 1, 3.0 + (frame - 75) * 8.5 / 25.0, 0.0) for frame in range(76, 151)
    ]
    report = match_statistics(walk + sprint, StatsConfig())
    zones = report["players"][0]["speed_zones"]
    assert zones["walk"]["distance_m"] > 1.0
    assert zones["sprint"]["distance_m"] > 10.0
    assert zones["sprint"]["distance_m"] > zones["walk"]["distance_m"]
    total = sum(zone["distance_m"] for zone in zones.values())
    assert abs(total - report["players"][0]["distance_m"]) < 1.5


def test_a_change_of_pace_is_counted_once_not_once_per_frame():
    slow = [_detection(frame, 1, frame * 0.5 / 25.0, 0.0) for frame in range(1, 51)]
    fast = [_detection(frame, 1, 1.0 + (frame - 50) * 8.0 / 25.0, 0.0) for frame in range(51, 126)]
    report = match_statistics(slow + fast, StatsConfig())
    player = report["players"][0]
    assert 1 <= player["accelerations"] <= 4
    assert player["peak_acceleration_m_s2"] >= 2.0
    assert player["top_speed_at"].startswith("00:0")


def test_team_shape_needs_enough_players_to_mean_anything():
    two = [_detection(frame, track, track * 5.0, 0.0) for frame in range(1, 26) for track in (1, 2)]
    assert match_statistics(two, StatsConfig())["teams"]["left"]["shape"] == {}

    five = [
        _detection(frame, track, track * 5.0, track * 2.0)
        for frame in range(1, 26)
        for track in range(1, 6)
    ]
    shape = match_statistics(five, StatsConfig())["teams"]["left"]["shape"]
    assert shape["width_m"] > 5.0
    assert shape["depth_m"] > 15.0
    assert shape["compactness_m"] > 0


def test_line_height_is_positive_when_a_team_is_camped_in_the_other_half():
    """The side defending the left goal attacks towards +x."""
    advanced = [
        _detection(frame, track, 30.0 + track, track * 2.0, team="left")
        for frame in range(1, 26)
        for track in range(1, 6)
    ]
    deep = [
        _detection(frame, track + 10, 30.0 + track, track * 2.0, team="right")
        for frame in range(1, 26)
        for track in range(1, 6)
    ]
    report = match_statistics(advanced + deep, StatsConfig())
    assert report["teams"]["left"]["shape"]["line_height_m"] > 20
    assert report["teams"]["right"]["shape"]["line_height_m"] < -20


def test_turnovers_count_the_ball_changing_teams():
    """The ball has to travel there: a teleport is rejected before it counts."""
    rows = []
    for frame in range(1, 81):
        rows.append(_detection(frame, 1, 0.0, 0.0, team="left"))
        rows.append(_detection(frame, 2, 40.0, 0.0, team="right"))
        if frame <= 20:
            ball_x = 0.4
        elif frame >= 70:
            ball_x = 39.6
        else:
            ball_x = 0.4 + (frame - 20) * (39.2 / 50)  # under 1 m per frame
        rows.append(_detection(frame, 99, ball_x, 0.0, role="ball", team=None))
    flow = match_statistics(rows, StatsConfig())["possession_flow"]
    assert flow["turnovers"] == 1
    assert flow["spells"] == 2
    assert flow["longest_spell_s"] > 0.5
