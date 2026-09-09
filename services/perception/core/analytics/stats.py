"""Per-player statistics, every number traceable to where it came from.

Counting statistics are read off the ledger and nowhere else, so any figure can
be expanded into the list of events that produced it.  Continuous quantities —
distance run, top speed, heatmaps — have no events to be read from; they come
from the tracking table and say so in `source`.  That is the one place this
implementation departs from "the ledger is the only source", and it does so
openly rather than by inventing an event per frame.

Every player also carries `coverage`, because a broadcast statistic is a
statistic over the frames the camera happened to show.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from . import taxonomy as tx

MAX_PLAUSIBLE_SPEED = 12.0    # m/s; above this it is a tracking error, not a player
MAX_GAP_FRAMES = 3            # do not accumulate distance across longer holes
SPRINT_SPEED = 7.0            # m/s
HIGH_INTENSITY_SPEED = 5.5
SPRINT_MIN_S = 0.8


def build(run, ledger, directions: dict) -> dict:
    events = ledger.events
    by_player = defaultdict(list)
    for event in events:
        if event.player_id:
            by_player[event.player_id].append(event)

    physical = _physical(run)
    spatial = _spatial(run, directions)

    players = {}
    for player_id, entry in run.players.items():
        own = by_player.get(player_id, [])
        players[player_id] = {
            "player_id": player_id,
            "team": entry.get("team"),
            "jersey": entry.get("jersey"),
            "name": entry.get("name"),
            "role": entry.get("role"),
            "attribution_confidence": round(entry.get("attribution_confidence", 0.0), 3),
            "track_ids": entry.get("track_ids", []),
            "coverage": _coverage(run, entry),
            "on_ball": _on_ball(own, run.fps),
            "passing": _passing(own),
            "defensive": _defensive(own),
            "physical": physical.get(player_id, _empty_physical()),
            "spatial": spatial.get(player_id, _empty_spatial()),
            "timeline": _timeline(own),
        }

    return {
        "meta": dict(run.meta, attacking_directions=directions),
        "identification": _identification(players, events),
        "teams": _team_totals(events, run),
        "passing_network": _network(events),
        "players": players,
    }


def _identification(players: dict, events) -> dict:
    """How much of the report is about a named player and how much is a fragment.

    Without a shirt number a track cannot be joined to the rest of that player's
    afternoon, so it stays its own anonymous row.  A reader who does not know
    what share of the events landed on such rows will overread the table.
    """
    named = {pid for pid, p in players.items()
             if p.get("jersey") and p.get("role") != "referee"}
    anonymous = {pid for pid, p in players.items()
                 if not p.get("jersey") and p.get("role") != "referee"}
    on_ball = [e for e in events if e.player_id and e.type in
               (tx.TOUCH, tx.CARRY, tx.PASS, tx.RECEPTION, tx.LOSS,
                tx.RECOVERY, tx.INTERCEPTION, tx.CHALLENGE, tx.SHOT)]
    attributed = sum(1 for e in on_ball if e.player_id in named)
    return {
        "identified_players": len(named),
        "anonymous_fragments": len(anonymous),
        "on_ball_events": len(on_ball),
        "attributed_to_identified": attributed,
        "attributed_share": round(attributed / len(on_ball), 3) if on_ball else None,
        "note": "anonymous fragments are tracks whose number was never read; "
                "they are kept separate rather than merged on a guess",
    }


# --- coverage ---------------------------------------------------------------
def _coverage(run, entry) -> dict:
    frames_visible = int(entry.get("frames", 0))
    total = max(1, int(run.meta.get("frames_total", 0)) + 1)
    return {
        "frames_visible": frames_visible,
        "time_visible_s": round(frames_visible / run.fps, 2),
        "share_of_clip": round(frames_visible / total, 3),
        "note": "visible on the main camera; not playing time",
    }


# --- ledger-derived ---------------------------------------------------------
def _on_ball(events, fps) -> dict:
    touches = [e for e in events if e.type == tx.TOUCH]
    carries = [e for e in events if e.type == tx.CARRY]
    shots = [e for e in events if e.type == tx.SHOT]
    receptions = [e for e in events if e.type == tx.RECEPTION]
    carry_distance = sum(e.attributes.get("distance_m", 0.0) for e in carries)
    return {
        "source": "ledger",
        "touches": len(touches),
        "touches_contested": sum(1 for e in touches if e.attributes.get("contested")),
        "time_on_ball_s": round(sum(e.duration_s for e in touches), 2),
        "receptions": len(receptions),
        "receptions_in_box": sum(1 for e in receptions if e.attributes.get("in_box")),
        "carries": len(carries),
        "carry_distance_m": round(carry_distance, 1),
        "carry_progressive_m": round(
            sum(max(0.0, e.attributes.get("progressive_m", 0.0)) for e in carries), 1),
        "progressive_carries": sum(1 for e in carries if e.attributes.get("progressive")),
        "shots": len(shots),
    }


def _passing(events) -> dict:
    passes = [e for e in events if e.type == tx.PASS]
    complete = [e for e in passes if e.outcome == tx.OUTCOME_COMPLETE]
    lengths = [e.attributes.get("length_m", 0.0) for e in passes]
    return {
        "source": "ledger",
        "attempted": len(passes),
        "completed": len(complete),
        "accuracy": round(len(complete) / len(passes), 3) if passes else None,
        "forward": sum(1 for e in passes if e.attributes.get("forward")),
        "progressive": sum(1 for e in passes if e.attributes.get("progressive")),
        "progressive_m": round(
            sum(max(0.0, e.attributes.get("progressive_m", 0.0)) for e in complete), 1),
        "into_final_third": sum(1 for e in passes if e.attributes.get("into_final_third")),
        "into_box": sum(1 for e in passes if e.attributes.get("into_box")),
        "avg_length_m": round(float(np.mean(lengths)), 2) if lengths else None,
        "long_passes": sum(1 for length in lengths if length >= 25.0),
        "total_length_m": round(float(np.sum(lengths)), 1) if lengths else 0.0,
    }


def _defensive(events) -> dict:
    return {
        "source": "ledger",
        "losses": sum(1 for e in events if e.type == tx.LOSS),
        "recoveries": sum(1 for e in events if e.type == tx.RECOVERY),
        "interceptions": sum(1 for e in events if e.type == tx.INTERCEPTION),
        "challenges": sum(1 for e in events if e.type == tx.CHALLENGE),
        "challenges_lost": sum(1 for e in events
                               if e.type == tx.CHALLENGE and e.outcome == "lost"),
    }


def _timeline(events) -> list:
    """Every event of this player, ready to be turned into clickable jumps."""
    out = []
    for event in sorted(events, key=lambda e: e.clock.frame):
        if event.type in (tx.POSSESSION_PHASE, tx.LOOSE_BALL, tx.OUT_OF_VIEW):
            continue
        out.append({
            "event_id": event.event_id,
            "type": event.type,
            "outcome": event.outcome,
            "frame": event.clock.frame,
            "video_time_s": event.clock.video_time_s,
            "match_clock": event.clock.match_clock,
            "confidence": event.confidence,
            "player_confidence": round(event.player_confidence, 3),
        })
    return out


def _team_totals(events, run) -> dict:
    phases = [e for e in events if e.type == tx.POSSESSION_PHASE]
    held = defaultdict(float)
    for phase in phases:
        held[phase.team] += phase.duration_s
    total = sum(held.values()) or 1.0
    out = {}
    for team in ("left", "right"):
        team_events = [e for e in events if e.team == team]
        passes = [e for e in team_events if e.type == tx.PASS]
        complete = [e for e in passes if e.outcome == tx.OUTCOME_COMPLETE]
        out[team] = {
            "possession_s": round(held.get(team, 0.0), 2),
            "possession_share": round(held.get(team, 0.0) / total, 3),
            "phases": sum(1 for p in phases if p.team == team),
            "touches": sum(1 for e in team_events if e.type == tx.TOUCH),
            "passes": len(passes),
            "pass_accuracy": round(len(complete) / len(passes), 3) if passes else None,
            "shots": sum(1 for e in team_events if e.type == tx.SHOT),
        }
    return out


def _network(events) -> list:
    pairs = Counter()
    for event in events:
        if event.type == tx.PASS and event.outcome == tx.OUTCOME_COMPLETE \
                and event.player_id and event.related_player_id:
            pairs[(event.player_id, event.related_player_id)] += 1
    return [{"from": a, "to": b, "passes": n}
            for (a, b), n in pairs.most_common()]


# --- tracking-derived -------------------------------------------------------
def _empty_physical() -> dict:
    return {"source": "tracking", "distance_m": 0.0, "avg_speed_ms": None,
            "top_speed_ms": None, "sprints": 0, "sprint_distance_m": 0.0,
            "high_intensity_distance_m": 0.0, "samples": 0}


def _empty_spatial() -> dict:
    return {"source": "tracking", "avg_x": None, "avg_y": None,
            "thirds": {}, "heatmap": [], "width_m": None, "depth_m": None}


def _physical(run) -> dict:
    """Distance and speed, accumulated only across frames that actually adjoin.

    Track fragments and camera cuts leave holes; bridging one would credit a
    player with a teleport.  Segments are therefore accumulated per track, and
    only between frames no further apart than MAX_GAP_FRAMES.
    """
    out = {}
    detections = run.detections.dropna(subset=["x", "y"])
    per_player = defaultdict(lambda: {"distance": 0.0, "speeds": [], "dt": [],
                                      "sprint": 0.0, "hi": 0.0, "runs": []})
    for track_id, group in detections.groupby("track_id"):
        player_id = run.track_to_player.get(int(track_id))
        if not player_id:
            continue
        group = group.sort_values("frame")
        frames = group.frame.to_numpy()
        # The pitch projection jitters by tens of centimetres frame to frame; a
        # standing defender differentiates into 10 m/s of imaginary sprinting.
        # A three-frame median costs nothing in real movement and removes it.
        xs = group.x.rolling(3, center=True, min_periods=1).median().to_numpy()
        ys = group.y.rolling(3, center=True, min_periods=1).median().to_numpy()
        gaps = np.diff(frames)
        step = np.hypot(np.diff(xs), np.diff(ys))
        dt = gaps / run.fps
        speed = np.divide(step, dt, out=np.zeros_like(step), where=dt > 0)
        # A single bad projection spikes one sample; a real sprint lasts many.
        # Reporting a top speed off a smoothed series keeps the peak honest.
        speed = pd.Series(speed).rolling(3, center=True, min_periods=1).mean().to_numpy()
        usable = (gaps <= MAX_GAP_FRAMES) & (speed <= MAX_PLAUSIBLE_SPEED)
        bucket = per_player[player_id]
        bucket["distance"] += float(step[usable].sum())
        bucket["speeds"].extend(speed[usable].tolist())
        bucket["dt"].extend(dt[usable].tolist())
        bucket["sprint"] += float(step[usable & (speed >= SPRINT_SPEED)].sum())
        bucket["hi"] += float(step[usable & (speed >= HIGH_INTENSITY_SPEED)].sum())
        bucket["runs"].append((frames[1:][usable], speed[usable]))

    for player_id, bucket in per_player.items():
        speeds = np.array(bucket["speeds"])
        sprints = 0
        for frames, speed in bucket["runs"]:
            sprints += _count_sprints(frames, speed, run.fps)
        out[player_id] = {
            "source": "tracking",
            "distance_m": round(bucket["distance"], 1),
            "avg_speed_ms": round(float(np.average(speeds, weights=bucket["dt"])), 2)
            if len(speeds) and sum(bucket["dt"]) > 0 else None,
            "top_speed_ms": round(float(np.percentile(speeds, 97)), 2)
            if len(speeds) >= 10 else (round(float(speeds.max()), 2) if len(speeds) else None),
            "sprints": sprints,
            "sprint_distance_m": round(bucket["sprint"], 1),
            "high_intensity_distance_m": round(bucket["hi"], 1),
            "samples": int(len(speeds)),
        }
    return out


def _count_sprints(frames: np.ndarray, speed: np.ndarray, fps: float) -> int:
    """A sprint is a stretch above SPRINT_SPEED lasting at least SPRINT_MIN_S."""
    if len(frames) == 0:
        return 0
    need = max(1, int(round(SPRINT_MIN_S * fps)))
    count, streak = 0, 0
    for i, fast in enumerate(speed >= SPRINT_SPEED):
        contiguous = i > 0 and frames[i] - frames[i - 1] <= MAX_GAP_FRAMES
        if fast and (streak == 0 or contiguous):
            streak += 1
            if streak == need:
                count += 1
        else:
            streak = 1 if fast else 0
    return count


def _spatial(run, directions: dict, cols: int = 6, rows: int = 5) -> dict:
    out = {}
    detections = run.detections.dropna(subset=["x", "y"])
    detections = detections[(detections.x.abs() <= tx.HALF_LENGTH + 2) &
                            (detections.y.abs() <= tx.HALF_WIDTH + 2)]
    for player_id, group in detections.groupby(detections.track_id.map(run.track_to_player)):
        if not player_id:
            continue
        xs, ys = group.x.to_numpy(), group.y.to_numpy()
        team = group.team.dropna()
        attacking_right = directions.get(team.iloc[0] if len(team) else None, True)
        thirds = Counter(tx.third(x, attacking_right) for x in xs)
        total = sum(thirds.values()) or 1
        heat = np.zeros((cols, rows), dtype=int)
        for x, y in zip(xs, ys):
            cx, ry = tx.zone(x, y, cols, rows)
            heat[cx, ry] += 1
        out[player_id] = {
            "source": "tracking",
            "avg_x": round(float(np.mean(xs)), 2),
            "avg_y": round(float(np.mean(ys)), 2),
            "thirds": {k: round(v / total, 3) for k, v in thirds.items()},
            "heatmap": heat.T.tolist(),
            "width_m": round(float(np.percentile(ys, 90) - np.percentile(ys, 10)), 1),
            "depth_m": round(float(np.percentile(xs, 90) - np.percentile(xs, 10)), 1),
        }
    return out


def to_frame(report: dict) -> pd.DataFrame:
    """Flatten the per-player report for CSV/Parquet export."""
    rows = []
    for player_id, player in report["players"].items():
        row = {"player_id": player_id, "team": player["team"],
               "jersey": player["jersey"], "name": player["name"],
               "role": player["role"],
               "attribution_confidence": player["attribution_confidence"]}
        for section in ("coverage", "on_ball", "passing", "defensive", "physical"):
            for key, value in player[section].items():
                if key in ("source", "note"):
                    continue
                row["%s_%s" % (section, key)] = value
        spatial = player["spatial"]
        row["avg_x"] = spatial.get("avg_x")
        row["avg_y"] = spatial.get("avg_y")
        rows.append(row)
    return pd.DataFrame(rows)
