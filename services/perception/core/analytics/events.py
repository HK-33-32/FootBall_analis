"""Derive ledger events from the ball trajectory and possession runs.

Everything here comes out of geometry.  A pass is not recognised by a model that
was shown a million passes; it is what a transition between two possession runs
of the same team looks like.  That makes the events auditable — each one can be
replayed at its frame and checked — and it means the ledger exists today rather
than after a research model is ported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import taxonomy as tx
from .ledger import Clock, Event, EventLedger

MIN_PASS_M = 6.0          # shorter transfers are a deflection, not a pass
                          # (6 m, not 3, from the same ground-truth calibration)
CARRY_MIN_M = 3.0
CARRY_MIN_S = 0.4
PROGRESSIVE_M = 5.0       # gain towards the opponent goal that counts as progress
SHOT_MIN_SPEED = 12.0     # m/s
SHOT_GOAL_BAND = 12.0     # |y| within which a ball crossing the goal line is a shot


class EventBuilder:
    def __init__(self, run, ball_track: pd.DataFrame, frames: pd.DataFrame,
                 possession_runs: list[dict], directions: dict,
                 kickoff_offset_s: float | None = None, period: int | None = None):
        self.run = run
        self.ball = ball_track.reset_index(drop=True)
        self.frames = frames.reset_index(drop=True)
        self.runs = possession_runs
        self.directions = directions
        self.kickoff_offset_s = kickoff_offset_s
        self.period = period
        self.fps = run.fps
        self.ledger = EventLedger(meta=dict(run.meta))
        self.ledger.players = run.players
        self._track_team = self._team_by_track()
        self._positions = self._player_positions()

    # --- helpers ------------------------------------------------------------
    def _team_by_track(self) -> dict:
        out = {}
        for track_id, group in self.run.detections.groupby("track_id"):
            teams = [t for t in group.team if t]
            out[int(track_id)] = max(set(teams), key=teams.count) if teams else None
        return out

    def clock(self, frame: int) -> Clock:
        segment_time = frame / self.fps
        video_time = self.run.start_s + segment_time
        match_clock = None
        if self.kickoff_offset_s is not None:
            elapsed = max(0.0, video_time - self.kickoff_offset_s)
            match_clock = "%02d:%02d" % (int(elapsed // 60), int(elapsed % 60))
        return Clock(frame=int(frame), video_time_s=round(video_time, 3),
                     segment_time_s=round(segment_time, 3),
                     match_clock=match_clock, period=self.period)

    def _player_positions(self):
        """Pitch position of every tracked player, by (track, frame).

        Event coordinates are taken from the *player*, not from the ball.  A
        player stands on the ground, so his projection onto the pitch is the
        one thing the homography is reliable about; the ball's is not, and a
        pass measured from the ball would take its length from a point that can
        be a hundred metres off the pitch while the ball is in the air.
        """
        rows = self.run.detections.dropna(subset=["x", "y"])
        return {(int(t), int(f)): (float(x), float(y))
                for t, f, x, y in zip(rows.track_id, rows.frame, rows.x, rows.y)}

    def position_of(self, track_id: int, frame: int, row: int):
        """Where an event happened: the player's spot, the ball's as a fallback."""
        place = self._positions.get((int(track_id), int(frame)))
        if place is not None:
            return place
        for offset in (1, -1, 2, -2, 3, -3):
            place = self._positions.get((int(track_id), int(frame) + offset))
            if place is not None:
                return place
        return self.ball_at(row)

    def ball_at(self, row: int):
        x = float(self.ball.x.iloc[row])
        y = float(self.ball.y.iloc[row])
        if not (np.isfinite(x) and np.isfinite(y)):
            return (0.0, 0.0)
        return x, y

    def player_of(self, track_id: int):
        return self.run.track_to_player.get(int(track_id))

    def confidence_of(self, track_id: int) -> float:
        player_id = self.player_of(track_id)
        if not player_id:
            return 0.0
        return float(self.run.players.get(player_id, {}).get("attribution_confidence", 0.0))

    def attacking_right(self, team) -> bool:
        return bool(self.directions.get(team, True))

    def _progress(self, team, start, end) -> float:
        """Metres gained towards the opponent goal."""
        gx, gy = tx.goal_centre(self.attacking_right(team))
        return float(np.hypot(gx - start[0], gy - start[1]) -
                     np.hypot(gx - end[0], gy - end[1]))

    def _emit(self, type_, frame, **kwargs) -> Event:
        event = Event(event_id=self.ledger.next_id(type_), type=type_,
                      clock=self.clock(frame), **kwargs)
        return self.ledger.add(event)

    # --- derivation ---------------------------------------------------------
    def build(self) -> EventLedger:
        self._on_ball_events()
        self._phases()
        self._coverage_gaps()
        self.ledger.sort()
        return self.ledger

    def _on_ball_events(self):
        held = [r for r in self.runs if r["track_id"] != -1]
        for index, run in enumerate(held):
            self._touch_and_carry(run)
            if index + 1 < len(held):
                self._transition(run, held[index + 1])
            else:
                self._trailing_shot(run)

    def _touch_and_carry(self, run):
        track_id = run["track_id"]
        team = self._track_team.get(track_id)
        start = self.position_of(track_id, run["start_frame"], run["start_row"])
        end = self.position_of(track_id, run["end_frame"], run["end_row"])
        self._emit(tx.TOUCH, run["start_frame"], team=team, track_id=track_id,
                   player_id=self.player_of(track_id),
                   player_confidence=self.confidence_of(track_id),
                   start_xy=start, duration_s=round(run["duration_s"], 3),
                   confidence=round(_touch_confidence(run), 3),
                   attributes={"contested": run["contested"],
                               "mean_distance_m": round(run["mean_dist"], 2),
                               "zone": tx.zone(*start)})

        distance = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        if distance >= CARRY_MIN_M and run["duration_s"] >= CARRY_MIN_S:
            progress = self._progress(team, start, end)
            self._emit(tx.CARRY, run["start_frame"], team=team, track_id=track_id,
                       player_id=self.player_of(track_id),
                       player_confidence=self.confidence_of(track_id),
                       start_xy=start, end_xy=end,
                       duration_s=round(run["duration_s"], 3),
                       confidence=round(_touch_confidence(run), 3),
                       attributes={"distance_m": round(distance, 2),
                                   "progressive_m": round(progress, 2),
                                   "progressive": progress >= PROGRESSIVE_M,
                                   "speed_ms": round(distance / max(run["duration_s"], 1e-3), 2)})

    def _transition(self, current, nxt):
        a, b = current["track_id"], nxt["track_id"]
        team_a, team_b = self._track_team.get(a), self._track_team.get(b)
        start = self.position_of(a, current["end_frame"], current["end_row"])
        end = self.position_of(b, nxt["start_frame"], nxt["start_row"])
        travel = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        gap_s = (nxt["start_frame"] - current["end_frame"]) / self.fps
        same_team = team_a is not None and team_a == team_b
        loose = self.frames.iloc[current["end_row"]:nxt["start_row"] + 1]
        contested = bool(loose.contested.any()) if len(loose) else False
        if nxt["start_row"] > current["end_row"]:
            observed = float(self.ball.observed.iloc[
                current["end_row"]:nxt["start_row"] + 1].mean())
        else:
            observed = 1.0
        confidence = round(min(1.0, 0.45 + 0.5 * observed), 3)

        common = dict(team=team_a, track_id=a, related_track_id=b,
                      player_id=self.player_of(a), related_player_id=self.player_of(b),
                      player_confidence=self.confidence_of(a),
                      start_xy=start, end_xy=end,
                      duration_s=round(gap_s, 3), confidence=confidence)

        player_a, player_b = self.player_of(a), self.player_of(b)
        if player_a and player_a == player_b:
            # the same player under two track ids: an occlusion split one spell
            # of control in two.  Crediting a pass here would invent a completed
            # pass and a reception out of a player never losing the ball.
            self._emit(tx.CARRY, current["end_frame"], team=team_a, track_id=a,
                       player_id=player_a, player_confidence=self.confidence_of(a),
                       start_xy=start, end_xy=end, duration_s=round(gap_s, 3),
                       confidence=round(confidence * 0.7, 3),
                       attributes={"distance_m": round(travel, 2),
                                   "progressive_m": round(
                                       self._progress(team_a, start, end), 2),
                                   "progressive": self._progress(team_a, start, end) >= PROGRESSIVE_M,
                                   "note": "bridged across a track split"})
            return

        if travel < MIN_PASS_M:
            # a shove, a rebound, a tackle: the owner changes with no ball flight
            self._emit(tx.CHALLENGE, current["end_frame"], **common,
                       outcome="lost" if not same_team else "kept",
                       attributes={"travel_m": round(travel, 2), "contested": contested})
            if not same_team:
                self._pair_loss_recovery(current, nxt, start, end, confidence,
                                         kind=tx.RECOVERY)
            return

        progress = self._progress(team_a, start, end)
        right_a = self.attacking_right(team_a)
        start_third = tx.third(start[0], right_a)
        end_third = tx.third(end[0], right_a)
        attributes = {
            "length_m": round(travel, 2),
            "progressive_m": round(progress, 2),
            "progressive": progress >= PROGRESSIVE_M,
            "forward": bool(progress > 0),
            "speed_ms": round(travel / max(gap_s, 1e-3), 2),
            "angle_deg": round(float(np.degrees(np.arctan2(end[1] - start[1],
                                                           end[0] - start[0]))), 1),
            "start_third": start_third,
            "end_third": end_third,
            "into_final_third": start_third != "attacking" and end_third == "attacking",
            "into_box": tx.in_penalty_box(end[0], end[1], right_a),
            "ball_observed_share": round(observed, 2),
            "contested": contested,
        }
        if not same_team and self._looks_like_shot(team_a, start, end, travel, gap_s):
            self._emit(tx.SHOT, current["end_frame"], team=team_a, track_id=a,
                       player_id=self.player_of(a),
                       player_confidence=self.confidence_of(a),
                       start_xy=start, end_xy=end, outcome=tx.OUTCOME_UNKNOWN,
                       confidence=round(confidence * 0.6, 3),
                       attributes=dict(attributes,
                                       note="geometry only; no goal detection"))
            return

        outcome = tx.OUTCOME_COMPLETE if same_team else tx.OUTCOME_INCOMPLETE
        self._emit(tx.PASS, current["end_frame"], **common, outcome=outcome,
                   attributes=attributes)
        if same_team:
            self._emit(tx.RECEPTION, nxt["start_frame"], team=team_b, track_id=b,
                       player_id=self.player_of(b),
                       related_player_id=self.player_of(a), related_track_id=a,
                       player_confidence=self.confidence_of(b),
                       start_xy=end, confidence=confidence,
                       attributes={"length_m": round(travel, 2),
                                   "in_box": tx.in_penalty_box(
                                       end[0], end[1], self.attacking_right(team_b))})
        else:
            self._pair_loss_recovery(current, nxt, start, end, confidence,
                                     kind=tx.INTERCEPTION)

    def _pair_loss_recovery(self, current, nxt, start, end, confidence, kind):
        a, b = current["track_id"], nxt["track_id"]
        self._emit(tx.LOSS, current["end_frame"], team=self._track_team.get(a),
                   track_id=a, player_id=self.player_of(a),
                   related_player_id=self.player_of(b), related_track_id=b,
                   player_confidence=self.confidence_of(a),
                   start_xy=start, confidence=confidence,
                   attributes={"third": tx.third(
                       start[0], self.attacking_right(self._track_team.get(a)))})
        self._emit(kind, nxt["start_frame"], team=self._track_team.get(b),
                   track_id=b, player_id=self.player_of(b),
                   related_player_id=self.player_of(a), related_track_id=a,
                   player_confidence=self.confidence_of(b),
                   start_xy=end, confidence=confidence,
                   attributes={"third": tx.third(
                       end[0], self.attacking_right(self._track_team.get(b)))})

    def _looks_like_shot(self, team, start, end, travel, gap_s) -> bool:
        if team is None or gap_s <= 0:
            return False
        if travel / gap_s < SHOT_MIN_SPEED:
            return False
        gx, _ = tx.goal_centre(self.attacking_right(team))
        towards = abs(end[0] - gx) < abs(start[0] - gx)
        near_goal = abs(end[0]) > tx.HALF_LENGTH - 12.0 and abs(end[1]) < SHOT_GOAL_BAND
        return bool(towards and near_goal and
                    tx.third(start[0], self.attacking_right(team)) == "attacking")

    def _trailing_shot(self, run):
        """Last run of the clip: the ball may simply have left the frame."""
        tail = self.ball.iloc[run["end_row"]:]
        if len(tail) < 3:
            return
        start = self.position_of(run["track_id"], run["end_frame"], run["end_row"])
        end = self.ball_at(len(self.ball) - 1)
        gap_s = (int(tail.frame.iloc[-1]) - run["end_frame"]) / self.fps
        travel = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        team = self._track_team.get(run["track_id"])
        if self._looks_like_shot(team, start, end, travel, gap_s):
            self._emit(tx.SHOT, run["end_frame"], team=team,
                       track_id=run["track_id"],
                       player_id=self.player_of(run["track_id"]),
                       player_confidence=self.confidence_of(run["track_id"]),
                       start_xy=start, end_xy=end, outcome=tx.OUTCOME_UNKNOWN,
                       confidence=0.35,
                       attributes={"length_m": round(travel, 2),
                                   "note": "clip ends before the outcome"})

    def _phases(self):
        """Team possession spells, so share-of-ball is read off the ledger."""
        current = None
        for run in self.runs:
            team = self._track_team.get(run["track_id"]) if run["track_id"] != -1 else None
            if team is None:
                if current:
                    self._close_phase(current, run["start_frame"])
                    current = None
                self._emit(tx.LOOSE_BALL, run["start_frame"],
                           duration_s=round(run["duration_s"], 3), confidence=0.8,
                           attributes={"frames": run["frames"]})
                continue
            if current and current["team"] == team:
                current["end_frame"] = run["end_frame"]
                current["touches"] += 1
                continue
            if current:
                self._close_phase(current, run["start_frame"])
            current = {"team": team, "start_frame": run["start_frame"],
                       "end_frame": run["end_frame"], "touches": 1}
        if current:
            self._close_phase(current, current["end_frame"])

    def _close_phase(self, phase, end_frame):
        duration = max(0.0, (end_frame - phase["start_frame"]) / self.fps)
        self._emit(tx.POSSESSION_PHASE, phase["start_frame"], team=phase["team"],
                   duration_s=round(duration, 3), confidence=0.85,
                   attributes={"touches": phase["touches"],
                               "end_frame": int(end_frame)})

    def _coverage_gaps(self):
        """Stretches with no usable pitch view: replays, close-ups, cuts.

        Statistics computed over a broadcast are statistics over what the camera
        showed.  Writing the blind stretches into the ledger is what lets the
        report say so, instead of quietly reporting a striker who ran 400 metres.
        """
        seen = set(int(f) for f in self.run.detections.frame.unique())
        if not seen:
            return
        first, last = min(seen), max(seen)
        gap_start = None
        for frame in range(first, last + 1):
            if frame in seen:
                if gap_start is not None and frame - gap_start >= self.fps:
                    self._emit(tx.OUT_OF_VIEW, gap_start, confidence=0.9,
                               duration_s=round((frame - gap_start) / self.fps, 3),
                               attributes={"end_frame": frame})
                gap_start = None
            elif gap_start is None:
                gap_start = frame
        if gap_start is not None and last - gap_start >= self.fps:
            self._emit(tx.OUT_OF_VIEW, gap_start, confidence=0.9,
                       duration_s=round((last - gap_start) / self.fps, 3),
                       attributes={"end_frame": last})


def _touch_confidence(run) -> float:
    """Closer and longer contact means a surer touch."""
    proximity = max(0.0, 1.0 - run["mean_dist"] / 3.0)
    persistence = min(1.0, run["frames"] / 4.0)
    return 0.35 + 0.45 * proximity + 0.2 * persistence
