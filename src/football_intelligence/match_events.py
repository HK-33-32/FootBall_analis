"""Reading football events out of who had the ball and where it went.

Everything here is derived from one sequence: the chain of possession spells,
each a player holding the ball for a stretch of frames. What happened between
two spells is decided by geometry and by the clock.

*A pass* is the ball moving from one player to a team-mate. *An interception*
is the same movement ending at an opponent who was nowhere near the passer.
*A tackle* is possession changing hands while the two players were close enough
to have contested it -- the ball barely moves, the owner does. The distinction
between the last two is the only thing separating a defender who read the play
from one who won the ball in a challenge, and it is a distance, not a guess.

*A shot* is the ball leaving a player fast, in the attacking half, on a line
that would cross the goal line inside the frame. *A goal* is the ball actually
arriving there -- and, when a score timeline is supplied, only a goal if the
scoreboard agrees. That confirmation matters more than it sounds: a ball that
passes behind the net looks identical from a single camera, and a tracker that
loses the ball at the moment it crosses is the normal case, not the exception.
An assist is then the pass that set up a confirmed goal.

The honest limits. Every event needs the ball, so coverage of the ball track is
the ceiling on all of it. A pass whose flight is never seen cannot be told from
a turnover. Deflections are attributed to whoever ends up with the ball. None
of this replaces a human tagger; it makes the tape searchable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Point = tuple[float, float]


@dataclass(frozen=True)
class EventConfig:
    """Thresholds separating one kind of event from another."""

    fps: float = 25.0
    # A spell shorter than this is not a touch. Two players contesting a loose
    # ball are each nearest to it on alternating frames, and without this the
    # flicker reads as a tackle every other frame: measured on broadcast
    # footage, six of the seven events it produced were the same two duels.
    min_spell_frames: int = 5
    # Frames of no owner that still belong to the same spell.
    spell_gap_frames: int = 12
    # Beyond this the ball was out of play or out of view: whatever happened
    # between the two owners is not one action.
    max_flight_s: float = 4.0
    min_pass_distance_m: float = 3.0
    # Close enough that the new owner could have taken the ball off the old one.
    duel_radius_m: float = 3.0
    shot_speed_m_s: float = 12.0
    goal_line_m: float = 52.5
    goal_half_width_m: float = 3.66
    # How far wide of the frame still counts as an attempt rather than a pass.
    off_target_margin_m: float = 5.0
    shot_min_attacking_x_m: float = 15.0
    goal_confirm_window_s: float = 15.0
    assist_window_s: float = 15.0
    carry_min_m: float = 4.0
    # Two players wrestling over one ball trade it several times before it
    # settles. Until it does, that is one duel, not a tackle answered by an
    # interception answered by another tackle.
    duel_window_s: float = 1.5
    # A foul is not visible in tracking; its consequence is. Play stops, the
    # ball sits still, and the referee is standing over it. All three together
    # are a candidate worth a human -- or a VLM -- looking at, and nothing
    # stronger than that is claimed.
    dead_ball_speed_m_s: float = 1.0
    dead_ball_min_s: float = 1.6
    referee_near_m: float = 12.0
    progressive_pass_m: float = 10.0
    final_third_m: float = 17.5


@dataclass
class Spell:
    """One player in control of the ball, from first touch to last."""

    track: int
    team: str | None
    start: int
    end: int
    start_point: Point
    end_point: Point


@dataclass
class Tally:
    """What one player did, counted."""

    touches: int = 0
    passes: int = 0
    passes_completed: int = 0
    passes_received: int = 0
    progressive_passes: int = 0
    passes_into_final_third: int = 0
    losses: int = 0
    interceptions: int = 0
    tackles: int = 0
    dispossessed: int = 0
    shots: int = 0
    shots_on_target: int = 0
    goals: int = 0
    assists: int = 0
    carries: int = 0
    carry_distance_m: float = 0.0
    possession_time_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__}
        payload["carry_distance_m"] = round(self.carry_distance_m, 1)
        payload["possession_time_s"] = round(self.possession_time_s, 2)
        payload["pass_accuracy"] = (
            round(self.passes_completed / self.passes, 3) if self.passes else None
        )
        return payload


@dataclass
class _Event:
    kind: str
    frame: int
    team: str | None
    player: int | None
    detail: dict[str, Any] = field(default_factory=dict)


def _timecode(frame: int, fps: float) -> str:
    seconds = frame / fps if fps else 0.0
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def _attacking_sign(team: str | None) -> float:
    """Towards which goal this team plays: the side defending the left goal
    attacks towards +x."""
    return 1.0 if team == "left" else -1.0


def build_spells(
    owner_frames: dict[int, list[int]],
    teams: dict[int, str | None],
    ball_at: dict[int, Point],
    player_at: dict[int, dict[int, Point]],
    config: EventConfig,
) -> list[Spell]:
    """Turn per-frame ball ownership into an ordered chain of spells."""
    owner_at: dict[int, int] = {}
    for track, frames in owner_frames.items():
        for frame in frames:
            owner_at[frame] = track

    # Runs of frames with one owner, then the flicker removed, then what is
    # left rejoined: a player who was briefly displaced by a moment of noise
    # never lost the ball.
    runs: list[list[int]] = []
    for frame in sorted(owner_at):
        track = owner_at[frame]
        if runs and runs[-1][0] == track and frame - runs[-1][2] <= config.spell_gap_frames:
            runs[-1][2] = frame
            continue
        runs.append([track, frame, frame])
    solid = [run for run in runs if run[2] - run[1] + 1 >= config.min_spell_frames]

    spells: list[Spell] = []
    for track, start, end in solid:
        if spells and spells[-1].track == track:
            spells[-1].end = end
            continue
        spells.append(
            Spell(
                track=track,
                team=teams.get(track),
                start=start,
                end=end,
                start_point=(0.0, 0.0),
                end_point=(0.0, 0.0),
            )
        )
    kept: list[Spell] = []
    for spell in spells:
        spell.start_point = player_at.get(spell.start, {}).get(
            spell.track, ball_at.get(spell.start, (0.0, 0.0))
        )
        spell.end_point = player_at.get(spell.end, {}).get(
            spell.track, ball_at.get(spell.end, (0.0, 0.0))
        )
        kept.append(spell)
    return kept


def _distance(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _shot_line(release: Point, arrival: Point, team: str | None, config: EventConfig):
    """Where a ball on this line would cross the goal line it is aimed at.

    Returns the crossing offset from the centre of the goal, or None when the
    ball is not travelling towards that goal at all.
    """
    sign = _attacking_sign(team)
    goal_x = sign * config.goal_line_m
    dx = arrival[0] - release[0]
    if dx * sign <= 0:
        return None
    remaining = goal_x - release[0]
    if remaining * sign <= 0:
        return 0.0  # already level with the goal line
    scale = remaining / dx
    return release[1] + (arrival[1] - release[1]) * scale


def _ball_speed(
    ball_frames: list[int], ball_points: list[Point], frame: int, config: EventConfig
) -> float:
    """Speed of the ball just after ``frame``, over a short window."""
    window = max(1, int(round(0.4 * config.fps)))
    for index, value in enumerate(ball_frames):
        if value >= frame:
            later = min(len(ball_frames) - 1, index + window)
            span = (ball_frames[later] - ball_frames[index]) / config.fps
            if span <= 0:
                return 0.0
            return _distance(ball_points[index], ball_points[later]) / span
    return 0.0


def find_stoppages(
    ball_frames: list[int],
    ball_points: list[Point],
    referee_at: dict[int, list[Point]],
    config: EventConfig,
) -> list[tuple[int, int, float]]:
    """Stretches where the ball sat still with an official close by.

    Returns (start, end, referee distance) per stoppage. This is a candidate
    detector, not a foul detector: a throw-in, a substitution and a booking all
    look the same from here.
    """
    if len(ball_frames) < 2:
        return []
    still: list[list[int]] = []
    for index in range(1, len(ball_frames)):
        span = (ball_frames[index] - ball_frames[index - 1]) / config.fps
        if span <= 0:
            continue
        speed = _distance(ball_points[index - 1], ball_points[index]) / span
        if speed > config.dead_ball_speed_m_s:
            continue
        if still and ball_frames[index - 1] - still[-1][1] <= config.spell_gap_frames:
            still[-1][1] = ball_frames[index]
        else:
            still.append([ball_frames[index - 1], ball_frames[index]])

    stoppages: list[tuple[int, int, float]] = []
    for start, end in still:
        if (end - start + 1) / config.fps < config.dead_ball_min_s:
            continue
        ball_here = dict(zip(ball_frames, ball_points, strict=True)).get(start)
        best = None
        for frame in range(start, end + 1):
            for point in referee_at.get(frame, ()):
                if ball_here is None:
                    continue
                gap = _distance(point, ball_here)
                best = gap if best is None else min(best, gap)
        if best is not None and best <= config.referee_near_m:
            stoppages.append((start, end, round(best, 1)))
    return stoppages


def _score_changes(timeline: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """Frames at which a side's score went up, from a scoreboard reading."""
    changes: list[tuple[int, str]] = []
    previous: dict[str, int] = {}
    for entry in sorted(timeline, key=lambda item: int(item.get("frame", 0))):
        frame = int(entry.get("frame", 0))
        for side in ("left", "right"):
            if side not in entry or entry[side] is None:
                continue
            value = int(entry[side])
            if side in previous and value > previous[side]:
                changes.append((frame, side))
            previous[side] = value
    return changes


def detect_events(
    owner_frames: dict[int, list[int]],
    teams: dict[int, str | None],
    ball_track: dict[str, Any],
    player_at: dict[int, dict[int, Point]],
    config: EventConfig | None = None,
    score_timeline: list[dict[str, Any]] | None = None,
    referee_at: dict[int, list[Point]] | None = None,
    cards: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Everything derivable from the possession chain, with per-player tallies."""
    config = config or EventConfig()
    ball_frames = list(ball_track.get("frames", []))
    ball_points = [tuple(point) for point in ball_track.get("points", [])]
    ball_at = dict(zip(ball_frames, ball_points, strict=True))

    spells = build_spells(owner_frames, teams, ball_at, player_at, config)
    tallies: dict[int, Tally] = defaultdict(Tally)
    events: list[_Event] = []

    for spell in spells:
        tally = tallies[spell.track]
        tally.touches += 1
        tally.possession_time_s += (spell.end - spell.start + 1) / config.fps
        carried = _distance(spell.start_point, spell.end_point)
        if carried >= config.carry_min_m:
            tally.carries += 1
            tally.carry_distance_m += carried
            events.append(
                _Event(
                    "carry",
                    spell.start,
                    spell.team,
                    spell.track,
                    {"distance_m": round(carried, 1), "duration_s": round(
                        (spell.end - spell.start + 1) / config.fps, 2)},
                )
            )

    # ---------------------------------------------------------------- shots
    shots: list[_Event] = []
    for index, spell in enumerate(spells):
        arrival_frame = spells[index + 1].start if index + 1 < len(spells) else None
        release = ball_at.get(spell.end, spell.end_point)
        speed = _ball_speed(ball_frames, ball_points, spell.end, config)
        if speed < config.shot_speed_m_s:
            continue
        sign = _attacking_sign(spell.team)
        if release[0] * sign < config.shot_min_attacking_x_m:
            continue
        after = [
            point
            for frame, point in zip(ball_frames, ball_points, strict=True)
            if spell.end < frame <= (arrival_frame or spell.end + int(3 * config.fps))
        ]
        if not after:
            continue
        offset = _shot_line(release, after[-1], spell.team, config)
        if offset is None:
            continue
        on_target = abs(offset) <= config.goal_half_width_m
        if not on_target and abs(offset) > config.goal_half_width_m + config.off_target_margin_m:
            continue
        tallies[spell.track].shots += 1
        if on_target:
            tallies[spell.track].shots_on_target += 1
        event = _Event(
            "shot",
            spell.end,
            spell.team,
            spell.track,
            {
                "on_target": on_target,
                "ball_speed_kmh": round(speed * 3.6, 1),
                "crossing_offset_m": round(float(offset), 2),
            },
        )
        shots.append(event)
        events.append(event)

    # ------------------------------------------------------- passes and duels
    last_duel: tuple[int, frozenset[int]] | None = None
    for first, second in zip(spells, spells[1:], strict=False):
        flight_s = (second.start - first.end) / config.fps
        if flight_s > config.max_flight_s:
            continue
        release = ball_at.get(first.end, first.end_point)
        receipt = ball_at.get(second.start, second.start_point)
        travelled = _distance(release, receipt)
        holders_apart = _distance(first.end_point, second.start_point)
        detail = {
            "from": first.track,
            "to": second.track,
            "distance_m": round(travelled, 1),
            "flight_s": round(flight_s, 2),
        }
        if first.team is not None and first.team == second.team:
            if travelled < config.min_pass_distance_m:
                continue
            tallies[first.track].passes += 1
            tallies[first.track].passes_completed += 1
            tallies[second.track].passes_received += 1
            sign = _attacking_sign(first.team)
            gained = (receipt[0] - release[0]) * sign
            if gained >= config.progressive_pass_m:
                tallies[first.track].progressive_passes += 1
                detail["progressive"] = True
            if (
                release[0] * sign < config.final_third_m
                and receipt[0] * sign >= config.final_third_m
            ):
                tallies[first.track].passes_into_final_third += 1
                detail["into_final_third"] = True
            events.append(_Event("pass", first.end, first.team, first.track, detail))
            continue
        if first.team is None or second.team is None:
            continue
        # The ball changed teams. Whether that was taken or given away is a
        # matter of how far apart the two players were when it happened.
        pair = frozenset({first.track, second.track})
        if (
            last_duel is not None
            and last_duel[1] == pair
            and second.start - last_duel[0] <= config.duel_window_s * config.fps
        ):
            continue  # the same duel, still going
        last_duel = (second.start, pair)
        tallies[first.track].losses += 1
        if holders_apart <= config.duel_radius_m and travelled < config.min_pass_distance_m:
            tallies[second.track].tackles += 1
            tallies[first.track].dispossessed += 1
            detail["holders_apart_m"] = round(holders_apart, 1)
            events.append(_Event("tackle", second.start, second.team, second.track, detail))
        else:
            tallies[second.track].interceptions += 1
            tallies[first.track].passes += 1
            events.append(_Event("interception", second.start, second.team, second.track, detail))

    # ------------------------------------------------- stoppages and cards
    stoppages = find_stoppages(ball_frames, ball_points, referee_at or {}, config)
    for start, end, gap in stoppages:
        events.append(
            _Event(
                "stoppage",
                start,
                None,
                None,
                {
                    "duration_s": round((end - start + 1) / config.fps, 2),
                    "referee_distance_m": gap,
                    "note": "мяч стоит, судья рядом — возможен фол",
                },
            )
        )
    for card in cards or []:
        events.append(
            _Event(
                "card",
                int(card.get("frame", 0)),
                card.get("team"),
                card.get("player"),
                {"colour": card.get("colour", "yellow"), "source": card.get("source", "vlm")},
            )
        )

    # ---------------------------------------------------------------- goals
    goals: list[_Event] = []
    reached_goal: list[tuple[int, str]] = []
    for frame, point in zip(ball_frames, ball_points, strict=True):
        if abs(point[0]) >= config.goal_line_m and abs(point[1]) <= config.goal_half_width_m:
            side = "left" if point[0] > 0 else "right"  # the team attacking that goal
            if not reached_goal or frame - reached_goal[-1][0] > config.fps:
                reached_goal.append((frame, side))

    changes = _score_changes(score_timeline or [])
    confirmed_by_score: set[int] = set()
    window = config.goal_confirm_window_s * config.fps

    def _scorer(frame: int, team: str | None) -> _Event | None:
        candidates = [
            shot for shot in shots if shot.team == team and 0 <= frame - shot.frame <= window
        ]
        if candidates:
            return candidates[-1]
        owners = [s for s in spells if s.team == team and 0 <= frame - s.end <= window]
        if owners:
            last = owners[-1]
            return _Event("shot", last.end, last.team, last.track, {"inferred": True})
        return None

    if score_timeline is not None:
        # The scoreboard is the referee's answer. When it is available it
        # decides, including when it says nothing happened: a ball passing
        # behind the net looks like a goal from one camera.
        for frame, side in changes:
            attacker = _scorer(frame, side)
            goals.append(
                _Event(
                    "goal",
                    frame,
                    side,
                    attacker.player if attacker else None,
                    {"confirmed_by": "score", "shot_frame": attacker.frame if attacker else None},
                )
            )
            if attacker is not None:
                confirmed_by_score.add(attacker.frame)
    else:
        for frame, side in reached_goal:
            attacker = _scorer(frame, side)
            goals.append(
                _Event(
                    "goal",
                    frame,
                    side,
                    attacker.player if attacker else None,
                    {
                        "confirmed_by": "ball_position",
                        "shot_frame": attacker.frame if attacker else None,
                    },
                )
            )

    for goal in goals:
        events.append(goal)
        if goal.player is not None:
            tallies[goal.player].goals += 1
        assist_window = config.assist_window_s * config.fps
        prior = [
            event
            for event in events
            if event.kind == "pass"
            and event.team == goal.team
            and event.player != goal.player
            and 0 <= goal.frame - event.frame <= assist_window
            and event.detail.get("to") == goal.player
        ]
        if prior:
            passer = prior[-1].player
            if passer is not None:
                tallies[passer].assists += 1
                events.append(
                    _Event(
                        "assist",
                        prior[-1].frame,
                        goal.team,
                        passer,
                        {"for_goal_at": goal.frame},
                    )
                )

    events.sort(key=lambda item: (item.frame, item.kind))
    by_kind: dict[str, int] = defaultdict(int)
    for event in events:
        by_kind[event.kind] += 1

    team_totals: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        members = [track for track, team in teams.items() if team == side]
        totals = Tally()
        for track in members:
            tally = tallies.get(track)
            if tally is None:
                continue
            for name in Tally.__dataclass_fields__:
                setattr(totals, name, getattr(totals, name) + getattr(tally, name))
        team_totals[side] = totals.as_dict()

    # Possession flow comes off the same chain the events do, so the page
    # cannot say nine turnovers above a list holding one.
    runs: list[list[Any]] = []
    for spell in spells:
        if spell.team not in ("left", "right"):
            continue
        if runs and runs[-1][0] == spell.team:
            runs[-1][2] = spell.end
            continue
        runs.append([spell.team, spell.start, spell.end])
    durations = [(end - start + 1) / config.fps for _, start, end in runs]

    return {
        "possession_flow": {
            "spells": len(runs),
            "turnovers": max(0, len(runs) - 1),
            "median_spell_s": round(float(np.median(durations)), 2) if durations else 0.0,
            "longest_spell_s": round(max(durations), 2) if durations else 0.0,
        },
        "events": [
            {
                "kind": event.kind,
                "frame": event.frame,
                "timecode": _timecode(event.frame, config.fps),
                "team": event.team,
                "player": event.player,
                **event.detail,
            }
            for event in events
        ],
        "counts": dict(sorted(by_kind.items())),
        "by_player": {str(track): tally.as_dict() for track, tally in sorted(tallies.items())},
        "by_team": team_totals,
        # Goals inside the footage analysed, which is not the same thing as
        # the score on the board: a clip can start at 1-0 and end there.
        "score": _score_from(goals),
        "scoreboard": _standing_score(score_timeline),
        "score_source": (
            "scoreboard" if score_timeline is not None else ("ball_position" if goals else "none")
        ),
        "possession_chain": len(spells),
        "stoppages": [
            {"frame": start, "timecode": _timecode(start, config.fps),
             "duration_s": round((end - start + 1) / config.fps, 2),
             "referee_distance_m": gap}
            for start, end, gap in stoppages
        ],
        "thresholds": {
            "shot_speed_kmh": round(config.shot_speed_m_s * 3.6, 1),
            "duel_radius_m": config.duel_radius_m,
            "min_pass_distance_m": config.min_pass_distance_m,
            "max_flight_s": config.max_flight_s,
        },
    }


def _standing_score(timeline: list[dict[str, Any]] | None) -> dict[str, int] | None:
    """The last score the board showed, however the clip started."""
    if not timeline:
        return None
    last = max(timeline, key=lambda item: int(item.get("frame", 0)))
    return {
        "left": int(last.get("left") or 0),
        "right": int(last.get("right") or 0),
        "frame": int(last.get("frame", 0)),
    }


def _score_from(goals: list[_Event]) -> dict[str, int]:
    score = {"left": 0, "right": 0}
    for goal in goals:
        if goal.team in score:
            score[goal.team] += 1
    return score


__all__ = ["EventConfig", "Spell", "Tally", "build_spells", "detect_events"]
