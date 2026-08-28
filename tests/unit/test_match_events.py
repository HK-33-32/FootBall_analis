from __future__ import annotations

from football_intelligence.match_events import EventConfig, detect_events

CONFIG = EventConfig(fps=25.0)


def _ball(points: dict[int, tuple[float, float]]) -> dict:
    frames = sorted(points)
    return {"frames": frames, "points": [points[frame] for frame in frames]}


def _line(start, end, first_frame, last_frame):
    """Ball moving steadily from start to end, one point per frame."""
    span = max(1, last_frame - first_frame)
    return {
        frame: (
            start[0] + (end[0] - start[0]) * (frame - first_frame) / span,
            start[1] + (end[1] - start[1]) * (frame - first_frame) / span,
        )
        for frame in range(first_frame, last_frame + 1)
    }


def test_a_ball_moving_between_team_mates_is_a_pass():
    owner = {1: list(range(1, 6)), 2: list(range(16, 21))}
    teams = {1: "left", 2: "left"}
    ball = _ball(_line((0.0, 0.0), (26.0, 0.0), 1, 20))
    player_at = {frame: {1: (0.0, 0.0), 2: (26.0, 0.0)} for frame in range(1, 21)}

    result = detect_events(owner, teams, ball, player_at, CONFIG)

    assert result["counts"]["pass"] == 1
    assert result["by_player"]["1"]["passes"] == 1
    assert result["by_player"]["1"]["passes_completed"] == 1
    assert result["by_player"]["1"]["pass_accuracy"] == 1.0
    assert result["by_player"]["2"]["passes_received"] == 1
    # towards the goal this team attacks, and far enough to be progressive
    assert result["by_player"]["1"]["progressive_passes"] == 1


def test_a_pass_reaching_an_opponent_far_away_is_an_interception():
    owner = {1: list(range(1, 6)), 2: list(range(16, 21))}
    teams = {1: "left", 2: "right"}
    ball = _ball(_line((0.0, 0.0), (20.0, 0.0), 1, 20))
    player_at = {frame: {1: (0.0, 0.0), 2: (20.0, 0.0)} for frame in range(1, 21)}

    result = detect_events(owner, teams, ball, player_at, CONFIG)

    assert result["counts"]["interception"] == 1
    assert "tackle" not in result["counts"]
    assert result["by_player"]["2"]["interceptions"] == 1
    assert result["by_player"]["1"]["losses"] == 1


def test_losing_the_ball_to_someone_standing_next_to_you_is_a_tackle():
    """The ball barely moves and the two players are within a challenge."""
    owner = {1: list(range(1, 6)), 2: list(range(7, 12))}
    teams = {1: "left", 2: "right"}
    ball = _ball(_line((0.0, 0.0), (1.0, 0.4), 1, 11))
    player_at = {frame: {1: (0.0, 0.0), 2: (1.2, 0.4)} for frame in range(1, 12)}

    result = detect_events(owner, teams, ball, player_at, CONFIG)

    assert result["counts"]["tackle"] == 1
    assert result["by_player"]["2"]["tackles"] == 1
    assert result["by_player"]["1"]["dispossessed"] == 1
    assert result["by_player"]["1"]["losses"] == 1


def test_two_players_alternating_on_a_loose_ball_is_not_a_tackle_every_frame():
    """Both are nearest to the ball on alternating frames; nothing happened."""
    owner = {1: [1, 3, 5, 7, 9, 11, 13, 15], 2: [2, 4, 6, 8, 10, 12, 14]}
    teams = {1: "left", 2: "right"}
    ball = _ball(_line((0.0, 0.0), (0.6, 0.2), 1, 15))
    player_at = {frame: {1: (0.0, 0.0), 2: (0.8, 0.2)} for frame in range(1, 16)}

    result = detect_events(owner, teams, ball, player_at, CONFIG)

    assert "tackle" not in result["counts"]
    assert "interception" not in result["counts"]


def test_a_ball_that_disappears_for_too_long_joins_nothing():
    owner = {1: list(range(1, 6)), 2: list(range(200, 206))}
    teams = {1: "left", 2: "left"}
    ball = _ball(
        {**_line((0.0, 0.0), (2.0, 0.0), 1, 6), **_line((30.0, 0.0), (31.0, 0.0), 200, 206)}
    )
    player_at = {frame: {1: (0.0, 0.0), 2: (30.0, 0.0)} for frame in range(1, 207)}

    result = detect_events(owner, teams, ball, player_at, CONFIG)

    assert "pass" not in result["counts"]


def test_a_fast_ball_aimed_between_the_posts_is_a_shot_on_target():
    owner = {1: list(range(1, 7))}
    teams = {1: "left"}
    # released at x=40 and travelling 12 m in half a second: about 86 km/h
    ball = _ball(_line((40.0, 0.0), (52.0, 0.5), 6, 18))
    player_at = {frame: {1: (40.0, 0.0)} for frame in range(1, 19)}

    result = detect_events(owner, teams, ball, player_at, CONFIG)

    assert result["counts"]["shot"] == 1
    shot = next(event for event in result["events"] if event["kind"] == "shot")
    assert shot["on_target"] is True
    assert result["by_player"]["1"]["shots_on_target"] == 1


def test_a_fast_ball_wide_of_the_frame_is_a_shot_off_target():
    owner = {1: list(range(1, 7))}
    teams = {1: "left"}
    ball = _ball(_line((40.0, 0.0), (52.0, 7.0), 6, 18))
    player_at = {frame: {1: (40.0, 0.0)} for frame in range(1, 19)}

    result = detect_events(owner, teams, ball, player_at, CONFIG)

    shot = next(event for event in result["events"] if event["kind"] == "shot")
    assert shot["on_target"] is False
    assert result["by_player"]["1"]["shots"] == 1
    assert result["by_player"]["1"]["shots_on_target"] == 0


def test_a_ball_in_the_net_counts_as_a_goal_when_no_scoreboard_is_given():
    owner = {1: list(range(1, 7))}
    teams = {1: "left"}
    ball = _ball(_line((40.0, 0.0), (53.0, 0.5), 6, 19))
    player_at = {frame: {1: (40.0, 0.0)} for frame in range(1, 20)}

    result = detect_events(owner, teams, ball, player_at, CONFIG)

    assert result["score"] == {"left": 1, "right": 0}
    assert result["score_source"] == "ball_position"
    goal = next(event for event in result["events"] if event["kind"] == "goal")
    assert goal["player"] == 1
    assert result["by_player"]["1"]["goals"] == 1


def test_the_scoreboard_overrules_the_ball_when_both_are_available():
    """A ball behind the net looks like a goal; the score says otherwise."""
    owner = {1: list(range(1, 7))}
    teams = {1: "left"}
    ball = _ball(_line((40.0, 0.0), (53.0, 0.5), 6, 19))
    player_at = {frame: {1: (40.0, 0.0)} for frame in range(1, 20)}
    flat_score = [{"frame": 1, "left": 0, "right": 0}, {"frame": 200, "left": 0, "right": 0}]

    result = detect_events(owner, teams, ball, player_at, CONFIG, score_timeline=flat_score)

    assert result["score"] == {"left": 0, "right": 0}
    assert result["score_source"] == "scoreboard"
    assert not [event for event in result["events"] if event["kind"] == "goal"]


def test_a_goal_the_tracker_never_saw_is_recovered_from_the_score():
    owner = {1: list(range(1, 6))}
    teams = {1: "left"}
    ball = _ball(_line((40.0, 0.0), (48.0, 0.5), 1, 10))
    player_at = {frame: {1: (40.0, 0.0)} for frame in range(1, 11)}
    score = [{"frame": 1, "left": 0, "right": 0}, {"frame": 60, "left": 1, "right": 0}]

    result = detect_events(owner, teams, ball, player_at, CONFIG, score_timeline=score)

    assert result["score"] == {"left": 1, "right": 0}
    goal = next(event for event in result["events"] if event["kind"] == "goal")
    assert goal["confirmed_by"] == "score"
    assert goal["player"] == 1


def test_the_pass_that_set_up_a_confirmed_goal_is_an_assist():
    owner = {1: list(range(1, 6)), 2: list(range(16, 22))}
    teams = {1: "left", 2: "left"}
    ball = _ball(
        {**_line((20.0, 0.0), (40.0, 0.0), 1, 20), **_line((40.0, 0.0), (53.0, 0.4), 21, 34)}
    )
    player_at = {frame: {1: (20.0, 0.0), 2: (40.0, 0.0)} for frame in range(1, 35)}
    score = [{"frame": 1, "left": 0, "right": 0}, {"frame": 60, "left": 1, "right": 0}]

    result = detect_events(owner, teams, ball, player_at, CONFIG, score_timeline=score)

    assert result["by_player"]["2"]["goals"] == 1
    assert result["by_player"]["1"]["assists"] == 1
    assert result["counts"]["assist"] == 1


def test_carrying_the_ball_is_counted_and_a_standing_touch_is_not():
    owner = {1: list(range(1, 30)), 2: list(range(60, 70))}
    teams = {1: "left", 2: "left"}
    ball = _ball(_line((0.0, 0.0), (12.0, 0.0), 1, 70))
    player_at = {
        frame: {1: (0.0 + 12.0 * min(frame, 29) / 29, 0.0), 2: (12.0, 0.0)}
        for frame in range(1, 71)
    }

    result = detect_events(owner, teams, ball, player_at, CONFIG)

    assert result["by_player"]["1"]["carries"] == 1
    assert result["by_player"]["1"]["carry_distance_m"] >= 4.0
    assert result["by_player"]["2"]["carries"] == 0


def test_team_totals_add_up_from_the_players():
    owner = {1: list(range(1, 6)), 2: list(range(16, 21)), 3: list(range(30, 36))}
    teams = {1: "left", 2: "left", 3: "right"}
    ball = _ball(_line((0.0, 0.0), (30.0, 0.0), 1, 35))
    player_at = {
        frame: {1: (0.0, 0.0), 2: (14.0, 0.0), 3: (30.0, 0.0)} for frame in range(1, 36)
    }

    result = detect_events(owner, teams, ball, player_at, CONFIG)

    assert result["by_team"]["left"]["passes"] >= 1
    assert result["by_team"]["right"]["interceptions"] == 1
    assert result["by_team"]["left"]["touches"] == 2


def test_a_duel_that_trades_the_ball_several_times_is_one_event():
    """The ball changes hands three times in a second; it is one contest."""
    owner = {
        1: list(range(1, 8)) + list(range(15, 22)),
        2: list(range(8, 15)) + list(range(22, 29)),
    }
    teams = {1: "left", 2: "right"}
    ball = _ball(_line((0.0, 0.0), (1.0, 0.3), 1, 29))
    player_at = {frame: {1: (0.0, 0.0), 2: (1.0, 0.3)} for frame in range(1, 30)}

    result = detect_events(owner, teams, ball, player_at, CONFIG)

    contests = result["counts"].get("tackle", 0) + result["counts"].get("interception", 0)
    assert contests == 1


def test_the_board_score_is_reported_apart_from_goals_inside_the_clip():
    """A clip can start at 1-0 and end at 1-0 with no goal in it."""
    owner = {1: list(range(1, 8))}
    teams = {1: "left"}
    ball = _ball(_line((0.0, 0.0), (4.0, 0.0), 1, 10))
    player_at = {frame: {1: (0.0, 0.0)} for frame in range(1, 11)}
    board = [{"frame": 1, "left": 1, "right": 0}, {"frame": 200, "left": 1, "right": 0}]

    result = detect_events(owner, teams, ball, player_at, CONFIG, score_timeline=board)

    assert result["score"] == {"left": 0, "right": 0}
    assert result["scoreboard"]["left"] == 1


def test_a_still_ball_with_the_referee_over_it_is_a_stoppage_candidate():
    owner = {1: list(range(1, 8))}
    teams = {1: "left"}
    ball = _ball({frame: (10.0, 0.0) for frame in range(1, 80)})
    player_at = {frame: {1: (10.0, 0.0)} for frame in range(1, 80)}
    referee_at = {frame: [(14.0, 2.0)] for frame in range(1, 80)}

    result = detect_events(owner, teams, ball, player_at, CONFIG, referee_at=referee_at)

    assert len(result["stoppages"]) == 1
    assert result["stoppages"][0]["referee_distance_m"] < 12.0
    assert result["counts"]["stoppage"] == 1


def test_a_still_ball_with_no_official_nearby_is_not_a_stoppage():
    owner = {1: list(range(1, 8))}
    teams = {1: "left"}
    ball = _ball({frame: (10.0, 0.0) for frame in range(1, 80)})
    player_at = {frame: {1: (10.0, 0.0)} for frame in range(1, 80)}
    referee_at = {frame: [(-40.0, 20.0)] for frame in range(1, 80)}

    result = detect_events(owner, teams, ball, player_at, CONFIG, referee_at=referee_at)

    assert result["stoppages"] == []


def test_a_moving_ball_is_never_a_stoppage():
    owner = {1: list(range(1, 8))}
    teams = {1: "left"}
    ball = _ball(_line((0.0, 0.0), (40.0, 0.0), 1, 80))
    player_at = {frame: {1: (0.0, 0.0)} for frame in range(1, 81)}
    referee_at = {frame: [(0.0, 1.0)] for frame in range(1, 81)}

    result = detect_events(owner, teams, ball, player_at, CONFIG, referee_at=referee_at)

    assert result["stoppages"] == []


def test_cards_that_were_read_are_carried_into_the_events():
    owner = {1: list(range(1, 8))}
    teams = {1: "left"}
    ball = _ball(_line((0.0, 0.0), (4.0, 0.0), 1, 10))
    player_at = {frame: {1: (0.0, 0.0)} for frame in range(1, 11)}
    cards = [{"frame": 40, "colour": "yellow", "team": "right", "source": "vlm"}]

    result = detect_events(owner, teams, ball, player_at, CONFIG, cards=cards)

    card = next(event for event in result["events"] if event["kind"] == "card")
    assert card["colour"] == "yellow"
    assert card["timecode"] == "00:01.600"
