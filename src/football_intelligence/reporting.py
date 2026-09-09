"""Evidence-aware match report inspired by the supplied RuStat report.

This is an independent metric implementation, not a RuStat index/xG replica.
Missing observations remain null; camera-derived events are always estimates.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReportConfig(BaseModel):
    """Resolved, serialized metric definitions; overrides accepted by the CLI."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)
    pitch_length_m: float = Field(default=105.0, gt=0)
    pitch_width_m: float = Field(default=68.0, gt=0)
    penalty_area_depth_m: float = Field(default=16.5, gt=0)
    penalty_area_width_m: float = Field(default=40.32, gt=0)
    short_pass_max_m: float = Field(default=10.0, gt=0)
    medium_pass_max_m: float = Field(default=40.0, gt=0)
    forward_cosine: float = Field(default=0.5, gt=0, le=1)
    progressive_own_half_m: float = Field(default=30.0, gt=0)
    progressive_cross_half_m: float = Field(default=15.0, gt=0)
    progressive_opponent_half_m: float = Field(default=10.0, gt=0)
    ppda_zone_depth_m: float = Field(default=40.0, gt=0)
    interval_ms: int = Field(default=900_000, gt=0)

    @model_validator(mode="after")
    def consistent_geometry(self):
        if self.short_pass_max_m >= self.medium_pass_max_m:
            raise ValueError("pass length boundaries must be increasing")
        if (
            self.penalty_area_depth_m >= self.pitch_length_m / 2
            or self.penalty_area_width_m > self.pitch_width_m
            or self.ppda_zone_depth_m > self.pitch_length_m
        ):
            raise ValueError("zone dimensions exceed the pitch")
        return self


class ReportEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    event_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    timestamp_ms: int = Field(ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    team: str | None = None
    player_id: str | None = None
    target_id: str | None = None
    outcome: bool | None = None
    position_m: tuple[float, float] | None = None
    end_position_m: tuple[float, float] | None = None
    attacking_sign: Literal[-1, 1] | None = None
    evidence_ids: list[str] = Field(min_length=1)
    source: str = Field(min_length=1)
    status: Literal["verified", "estimated"] = "estimated"
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_kind: Literal["calibrated", "model_score", "heuristic"] | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    qualifiers: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ordered_time(self):
        if self.end_ms is not None and self.end_ms < self.timestamp_ms:
            raise ValueError("end_ms precedes timestamp_ms")
        if any(not evidence.strip() for evidence in self.evidence_ids):
            raise ValueError("empty evidence reference")
        json.dumps(self.qualifiers, allow_nan=False)
        json.dumps(self.provenance, allow_nan=False)
        for key, value in self.qualifiers.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite qualifier: {key}")
        for key in ("distance_m", "xg", "xa"):
            value = self.qualifiers.get(key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int | float) or value < 0
            ):
                raise ValueError(f"{key} must be a nonnegative number")
        for key in ("xg", "xa"):
            if self.qualifiers.get(key, 0) is not None and self.qualifiers.get(key, 0) > 1:
                raise ValueError(f"{key} must be at most 1 per event")
        if self.confidence is not None and self.confidence_kind is None:
            raise ValueError("confidence requires confidence_kind")
        return self


class Period(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered_time(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("period must have positive duration")
        return self


class ReportAnnotations(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0.0"] = "1.0.0"
    match_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    teams: list[str] = Field(default_factory=list)
    events: list[ReportEvent] = Field(default_factory=list)
    # Complete types certify negative evidence over the declared interval.
    complete_event_types: list[str] = Field(default_factory=list)
    coverage_start_ms: int = Field(default=0, ge=0)
    coverage_end_ms: int = Field(gt=0)
    periods: list[Period] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_evidence(self):
        if self.coverage_end_ms <= self.coverage_start_ms:
            raise ValueError("invalid annotation coverage interval")
        ids = [event.event_id for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate event_id")
        if any(
            not self.coverage_start_ms <= e.timestamp_ms < self.coverage_end_ms
            or (e.end_ms is not None and e.end_ms > self.coverage_end_ms)
            for e in self.events
        ):
            raise ValueError("event outside annotation coverage")
        if self.teams and any(e.team and e.team not in self.teams for e in self.events):
            raise ValueError("event team is not declared in teams")
        periods = sorted(self.periods, key=lambda p: p.start_ms)
        for p in periods:
            if p.start_ms < self.coverage_start_ms or p.end_ms > self.coverage_end_ms:
                raise ValueError("period outside annotation coverage")
        if any(a.end_ms > b.start_ms for a, b in zip(periods, periods[1:], strict=False)):
            raise ValueError("overlapping periods")
        spells = sorted(
            (e for e in self.events if e.kind == "possession"), key=lambda e: e.timestamp_ms
        )
        if any(
            a.end_ms is not None and a.end_ms > b.timestamp_ms
            for a, b in zip(spells, spells[1:], strict=False)
        ):
            raise ValueError("possession spells must not overlap")
        return self


# (metric key, event type, qualifier). All fields are populated from evidence;
# this catalog also declares unavailable fields explicitly in a stable schema.
CATALOG = {
    "general": [
        ("goals", "goal", None),
        ("assists", "assist", None),
        ("chances", "chance", None),
        ("saves", "save", None),
        ("fouls", "foul", None),
        ("fouls_suffered", "foul_suffered", None),
        ("offsides", "offside", None),
        ("corners", "corner", None),
        ("yellow_cards", "card", "yellow"),
        ("red_cards", "card", "red"),
    ],
    "shots": [
        ("total", "shot", None),
        ("on_target", "shot", "on_target"),
        ("off_target", "shot", "off_target"),
        ("blocked", "shot", "blocked"),
        ("inside_box", "shot", "inside_box"),
        ("headers", "shot", "header"),
        ("free_kicks", "shot", "free_kick"),
        ("penalties", "shot", "penalty"),
    ],
    "passes": [
        ("total", "pass", None),
        ("short", "pass", "short"),
        ("medium", "pass", "medium"),
        ("long", "pass", "long"),
        ("forward", "pass", "forward"),
        ("lateral", "pass", "lateral"),
        ("backward", "pass", "backward"),
        ("progressive", "pass", "progressive"),
        ("into_final_third", "pass", "into_final_third"),
        ("into_box", "pass", "into_box"),
        ("key", "pass", "key"),
        ("crosses", "pass", "cross"),
        ("open_play", "pass", "open_play"),
        ("set_piece", "pass", "set_piece"),
        ("throw_ins", "pass", "throw_in"),
        ("corner_deliveries", "pass", "corner"),
    ],
    "duels": [
        ("total", "duel", None),
        ("aerial", "duel", "aerial"),
        ("ground", "duel", "ground"),
        ("attacking", "duel", "attacking"),
        ("defending", "duel", "defending"),
        ("dribbles", "dribble", None),
        ("tackles", "tackle", None),
    ],
    "recoveries_losses": [
        ("losses", "loss", None),
        ("losses_own_half", "loss", "own_half"),
        ("recoveries", "recovery", None),
        ("recoveries_opponent_half", "recovery", "opponent_half"),
        ("interceptions", "interception", None),
        ("interceptions_opponent_half", "interception", "opponent_half"),
        ("loose_ball_recoveries", "recovery", "loose_ball"),
        ("miscontrols", "miscontrol", None),
        ("clearances", "clearance", None),
    ],
    "attacks": [
        ("total", "attack", None),
        ("positional", "attack", "positional"),
        ("fast", "attack", "fast"),
        ("set_piece", "attack", "set_piece"),
        ("with_shot", "attack", "with_shot"),
        ("left_flank", "attack", "left_flank"),
        ("centre", "attack", "centre"),
        ("right_flank", "attack", "right_flank"),
    ],
    "pressing": [
        ("total", "pressure", None),
        ("high", "pressure", "high"),
        ("low", "pressure", "low"),
        ("build_ups", "build_up", None),
        ("unopposed_build_ups", "build_up", "unopposed"),
    ],
    "progression": [
        ("carries", "carry", None),
        ("final_third_entries", "entry", "final_third"),
        ("opponent_half_entries", "entry", "opponent_half"),
        ("box_entries", "entry", "box"),
    ],
    "possession": [
        ("spells", "possession", None),
        ("duration_0_10_s", "possession", "duration_0_10"),
        ("duration_10_20_s", "possession", "duration_10_20"),
        ("duration_20_45_s", "possession", "duration_20_45"),
        ("duration_over_45_s", "possession", "duration_over_45"),
    ],
}


def _qualifies(event: ReportEvent, key: str | None, config: ReportConfig) -> bool | None:
    if key is None:
        return True
    if key in event.qualifiers:
        value = event.qualifiers[key]
        return value if isinstance(value, bool) else None
    q = event.qualifiers
    start, end, sign = event.position_m, event.end_position_m, event.attacking_sign
    if key in {"yellow", "red"}:
        return q.get("colour") == key if "colour" in q else None
    if key.startswith("duration_"):
        if event.end_ms is None:
            return None
        duration = (event.end_ms - event.timestamp_ms) / 1000
        low, high = {
            "duration_0_10": (0, 10),
            "duration_10_20": (10, 20),
            "duration_20_45": (20, 45),
            "duration_over_45": (45, math.inf),
        }[key]
        return low <= duration < high
    if key in {"own_half", "opponent_half"} and start is not None and sign:
        return start[0] * sign < 0 if key == "own_half" else start[0] * sign >= 0
    if key == "inside_box" and start is not None and sign:
        return (
            config.pitch_length_m / 2 - config.penalty_area_depth_m
            <= start[0] * sign
            <= config.pitch_length_m / 2
            and abs(start[1]) <= config.penalty_area_width_m / 2
        )
    if event.kind == "pass":
        distance = q.get("distance_m")
        if start is not None and end is not None:
            distance = math.dist(start, end)
        if key in {"short", "medium", "long"} and distance is not None:
            return {
                "short": distance < config.short_pass_max_m,
                "medium": config.short_pass_max_m <= distance <= config.medium_pass_max_m,
                "long": distance > config.medium_pass_max_m,
            }[key]
        if start is not None and end is not None and sign:
            delta = (end[0] - start[0]) * sign
            cosine = delta / max(math.dist(start, end), 1e-12)
            if key in {"forward", "lateral", "backward"}:
                return {
                    "forward": cosine >= config.forward_cosine,
                    "backward": cosine <= -config.forward_cosine,
                    "lateral": -config.forward_cosine < cosine < config.forward_cosine,
                }[key]
            if key == "progressive":
                threshold = (
                    config.progressive_own_half_m
                    if end[0] * sign < 0
                    else config.progressive_cross_half_m
                    if start[0] * sign < 0
                    else config.progressive_opponent_half_m
                )
                return cosine >= config.forward_cosine and delta >= threshold
            if key == "into_final_third":
                return end[0] * sign >= config.pitch_length_m / 6
            if key == "into_box":
                return (
                    config.pitch_length_m / 2 - config.penalty_area_depth_m
                    <= end[0] * sign
                    <= config.pitch_length_m / 2
                    and abs(end[1]) <= config.penalty_area_width_m / 2
                )
    return None


def _metric(
    events: list[ReportEvent],
    kind: str,
    qualifier: str | None,
    complete: set[str],
    config: ReportConfig,
) -> dict:
    candidates = [e for e in events if e.kind == kind]
    selected, missing = [], 0
    for e in candidates:
        flag = _qualifies(e, qualifier, config)
        if flag is None:
            missing += 1
        elif flag:
            selected.append(e)
    available = bool(candidates) or kind in complete
    value = len(selected) if available and missing == 0 else None
    outcomes = [e.outcome for e in selected]
    successful = sum(v is True for v in outcomes) if outcomes and None not in outcomes else None
    status = (
        "unavailable"
        if value is None
        else "verified"
        if kind in complete and all(e.status == "verified" for e in candidates)
        else "estimated"
        if any(e.status == "estimated" for e in candidates)
        else "partial"
    )
    return {
        "value": value,
        "observed_count": len(selected),
        "successful": successful,
        "success_pct": round(successful * 100 / value, 2)
        if successful is not None and value
        else None,
        "status": status,
        "unit": "count",
        "unknown_qualifiers": missing,
        "evidence_ids": [e.event_id for e in selected],
        "reason": "missing event coverage or qualifiers" if value is None else None,
    }


def _sum_measure(events, kind, key, complete, unit, config):
    candidates = [e for e in events if e.kind == kind]
    values = [e.qualifiers.get(key) for e in candidates]
    known = [
        v
        for v in values
        if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
    ]
    base = _metric(events, kind, None, complete, config)
    value = sum(known) if len(known) == len(values) and base["value"] is not None else None
    return {
        "value": round(value, 4) if value is not None else None,
        "unit": unit,
        "status": base["status"] if value is not None else "unavailable",
        "evidence_ids": [e.event_id for e in candidates],
        "reason": None if value is not None else f"requires {key} for every {kind}",
    }


def _aggregate(events, complete, config):
    groups = {
        group: {
            name: _metric(events, kind, qualifier, complete, config)
            for name, kind, qualifier in specs
        }
        for group, specs in CATALOG.items()
    }
    for key in ("xg", "xa"):
        kind = "shot" if key == "xg" else "pass"
        # External xG/xA must carry its own model identity, never an invented formula.
        eligible = [
            e.model_copy(
                update={
                    "qualifiers": {
                        **e.qualifiers,
                        key: e.qualifiers.get(key) if e.qualifiers.get(f"{key}_model") else None,
                    }
                }
            )
            for e in events
        ]
        groups["shots" if key == "xg" else "passes"][key] = _sum_measure(
            eligible, kind, key, complete, "expected_goals", config
        )
    for kind, group in (("pass", "passes"), ("shot", "shots")):
        measured = []
        for e in events:
            if e.kind != kind:
                continue
            distance = e.qualifiers.get("distance_m") if kind == "pass" else None
            if e.position_m is not None:
                if kind == "pass" and e.end_position_m is not None:
                    distance = math.dist(e.position_m, e.end_position_m)
                elif kind == "shot" and e.attacking_sign:
                    distance = math.dist(
                        e.position_m, (e.attacking_sign * config.pitch_length_m / 2, 0)
                    )
            measured.append(e.model_copy(update={"qualifiers": {"distance_m": distance}}))
        total = _sum_measure(measured, kind, "distance_m", complete, "m", config)
        groups[group]["average_distance_m"] = {
            **total,
            "value": round(total["value"] / len(measured), 3)
            if total["value"] is not None and measured
            else None,
            "status": total["status"] if measured else "unavailable",
            "reason": total["reason"] if measured else "no observed events",
        }
    groups["general"]["provider_index"] = {
        "value": None,
        "status": "unavailable",
        "reason": "proprietary provider formula",
    }
    groups["pressing"]["ppda"] = {
        "value": None,
        "status": "unavailable",
        "reason": "requires complete spatial pass and defensive-action coverage",
    }
    groups["progression"]["carry_distance_m"] = _sum_measure(
        events, "carry", "distance_m", complete, "m", config
    )
    spells = [
        e.model_copy(
            update={
                "qualifiers": {
                    **e.qualifiers,
                    "duration_s": (e.end_ms - e.timestamp_ms) / 1000
                    if e.end_ms is not None
                    else None,
                }
            }
        )
        for e in events
        if e.kind == "possession"
    ]
    groups["possession"]["time_s"] = _sum_measure(
        spells, "possession", "duration_s", complete, "s", config
    )
    duration = groups["possession"]["time_s"]
    groups["possession"]["average_duration_s"] = {
        **duration,
        "value": round(duration["value"] / len(spells), 3)
        if duration["value"] is not None and spells
        else None,
        "status": duration["status"] if spells else "unavailable",
        "reason": duration["reason"] if spells else "no observed possession spells",
    }
    return groups


def legacy_report_events(
    report: dict, source: str, predictions: list[dict] | None = None
) -> list[ReportEvent]:
    """Adapt geometric candidates without upgrading them to semantic truth."""
    events = []
    fps = float(report["fps"])
    # Player feet on the ground, never a projected airborne-ball position.
    locations = {}
    for detection in predictions or []:
        pitch = detection.get("bbox_pitch") or {}
        if (
            (detection.get("attributes") or {}).get("role") != "ball"
            and pitch.get("x_bottom_middle") is not None
            and pitch.get("y_bottom_middle") is not None
        ):
            locations[(int(detection["frame"]), str(detection.get("track_id")))] = (
                float(pitch["x_bottom_middle"]),
                float(pitch["y_bottom_middle"]),
            )
    for raw in report.get("events", []):
        canonical = json.dumps(raw, sort_keys=True, ensure_ascii=False)
        event_id = "event-" + hashlib.sha256((source + canonical).encode()).hexdigest()[:20]
        kind = raw["kind"]
        timestamp = max(0, round((raw["frame"] - 1) / fps * 1000))
        duration = raw.get("duration_s")
        position = locations.get((raw["frame"], str(raw.get("player"))))
        recipient_frame = raw["frame"] + round(raw.get("flight_s", 0) * fps)
        destination = (
            locations.get((recipient_frame, str(raw.get("to")))) if kind == "pass" else None
        )
        events.append(
            ReportEvent(
                event_id=event_id,
                kind=kind,
                timestamp_ms=timestamp,
                end_ms=timestamp + round(duration * 1000) if duration is not None else None,
                team=raw.get("team"),
                player_id=str(raw["player"]) if raw.get("player") is not None else None,
                target_id=str(raw["to"]) if kind == "pass" and raw.get("to") is not None else None,
                position_m=position,
                end_position_m=destination,
                outcome=None,
                evidence_ids=[f"{source}#frame={raw['frame']}"],
                source="geometry-possession-chain",
                qualifiers={
                    "source_frame": raw["frame"],
                    "position_basis": "player_foot_proxy" if position else None,
                    "observed_teammate_transition": kind == "pass",
                    **{
                        k: v
                        for k, v in raw.items()
                        if k
                        not in {
                            "kind",
                            "frame",
                            "player",
                            "team",
                            "timecode",
                            "progressive",
                            "into_final_third",
                        }
                    },
                },
            )
        )
    return events


def report_with_duration(report: dict, duration_ms: int | None = None) -> dict:
    """Use source duration for coverage without mutating the caller's statistics."""
    duration_ms = duration_ms if duration_ms is not None else report.get("source_duration_ms")
    result = {**report, "coverage_basis": "observed_prediction_span",
              "source_duration_ms": duration_ms}
    if duration_ms is None:
        return result
    if duration_ms <= 0 or not math.isfinite(duration_ms):
        raise ValueError("source duration must be positive and finite")
    frames = math.ceil(duration_ms / 1000 * report["fps"] - 1e-9)
    if frames <= 0 or report.get("last_frame", 0) > frames:
        raise ValueError("source duration excludes observed frames")
    result.update(coverage_basis="source_duration", clip_frames=frames)
    count = report.get("frames_with_game_state")
    if count is not None and not 0 <= count <= frames:
        raise ValueError("observed frame count exceeds the source interval")
    result["coverage"] = round(count / frames, 4) if count is not None else None
    result["players"] = [
        {**player, "coverage": round(player["samples_used"] / frames, 4)
         if player.get("samples_used") is not None else None}
        for player in report.get("players", [])
    ]
    return result


def detailed_report(
    report: dict,
    *,
    annotations: ReportAnnotations | None = None,
    match_id: str = "",
    duration_ms: int | None = None,
    config: ReportConfig | None = None,
    predictions: list[dict] | None = None,
) -> dict:
    """Build team/player totals, period splits, pass networks and event maps."""
    report = report_with_duration(report, duration_ms)
    duration_ms = report["source_duration_ms"]
    config = config or ReportConfig()
    interval_ms = config.interval_ms
    source = str(
        report.get("video_source")
        or report.get("source")
        or match_id
        or report.get("title")
        or "predictions"
    )
    if annotations:
        if match_id and annotations.match_id != match_id:
            raise ValueError("annotation match_id does not match report")
        events = annotations.events
        complete = set(annotations.complete_event_types)
        start, end = annotations.coverage_start_ms, annotations.coverage_end_ms
        periods = annotations.periods
    else:
        events = legacy_report_events(report, source, predictions)
        complete = set()
        start = 0 if duration_ms else max(
            0, round((report.get("first_frame", 1) - 1) / report["fps"] * 1000)
        )
        end = duration_ms or round(report.get("last_frame", 0) / report["fps"] * 1000)
        periods = []
    end = max(end, start + 1)
    by_team, by_player = defaultdict(list), defaultdict(list)
    for event in events:
        if event.team:
            by_team[event.team].append(event)
        if event.player_id is not None:
            by_player[event.player_id].append(event)
    teams = (
        (set(annotations.teams) or set(by_team)) if annotations else set(report.get("teams", {}))
    )
    teams |= set(by_team)
    team_complete = {
        kind for kind in complete if all(e.team is not None for e in events if e.kind == kind)
    }
    totals = {team: _aggregate(by_team[team], team_complete, config) for team in sorted(teams)}
    # PPDA follows the reference's 40 m defensive zone; evaluate only when
    # the event provider explicitly certifies complete relevant coverage.
    required = {"pass", "tackle", "interception", "duel", "foul"}
    if required <= team_complete and len(teams) == 2:
        for team in teams:
            own = [e for e in by_team[team] if e.kind in required - {"pass"}]
            other = [e for e in events if e.team != team and e.team in teams and e.kind == "pass"]
            if all(e.position_m is not None and e.attacking_sign for e in own + other):
                boundary = config.pitch_length_m / 2 - config.ppda_zone_depth_m
                defensive = [e for e in own if e.position_m[0] * e.attacking_sign >= boundary]
                passes = [
                    e
                    for e in other
                    if e.outcome is True and e.position_m[0] * e.attacking_sign <= -boundary
                ]
                if all(e.outcome is not None for e in other):
                    totals[team]["pressing"]["ppda"] = {
                        "value": round(len(passes) / len(defensive), 3) if defensive else None,
                        "numerator": len(passes),
                        "denominator": len(defensive),
                        "status": "unavailable"
                        if not defensive
                        else "verified"
                        if all(e.status == "verified" for e in own + other)
                        else "estimated",
                        "unit": "passes_per_defensive_action",
                        "reason": None if defensive else "zero defensive-action denominator",
                        "evidence_ids": [e.event_id for e in passes + defensive],
                    }
    players = []
    recipients = defaultdict(list)
    for event in events:
        if (
            event.kind == "pass"
            and event.target_id
            and (event.outcome is True or event.qualifiers.get("observed_teammate_transition"))
        ):
            recipients[event.target_id].append(event)
    player_complete = {
        kind for kind in complete if all(e.player_id is not None for e in events if e.kind == kind)
    }
    recipient_complete = player_complete.copy()
    if any(e.kind == "pass" and e.outcome is True and not e.target_id for e in events):
        recipient_complete.discard("pass")
    # External roster IDs must not be conflated with internal track identities.
    known_players = (
        {}
        if annotations
        else {
            str(p["identity"]): p for p in report.get("players", []) if p.get("role") != "referee"
        }
    )
    for player_id in sorted(set(known_players) | set(by_player) | set(recipients)):
        p = known_players.get(player_id, {})
        player_events = by_player[player_id]
        related = player_events or recipients[player_id]
        players.append(
            {
                "player_id": player_id,
                "team": p.get("team") or (related[0].team if related else None),
                "jersey_number": p.get("jersey"),
                "identity_scope": "annotation_provider" if annotations else "track_or_linked_track",
                "minutes_played": None,
                "time_on_camera_s": p.get("time_on_camera_s"),
                "coverage": p.get("coverage"),
                "metrics": _aggregate(player_events, player_complete, config),
                "physical_visible_field": {
                    k: p[k]
                    for k in (
                        "distance_m",
                        "top_speed_kmh",
                        "speed_zones",
                        "accelerations",
                        "decelerations",
                        "mean_position",
                        "heatmap",
                    )
                    if k in p
                },
            }
        )
        players[-1]["metrics"]["passes"]["received"] = _metric(
            recipients[player_id], "pass", None, recipient_complete, config
        )
    windows = periods or [
        Period(
            name=f"video_{offset // interval_ms}",
            start_ms=offset,
            end_ms=min(offset + interval_ms, end),
        )
        for offset in range(start, end, interval_ms)
    ]
    bins = [
        {
            **p.model_dump(),
            "clock": "source_media_ms",
            "teams": {
                team: _aggregate(_window_events(by_team[team], p), team_complete, config)
                for team in sorted(teams)
            },
        }
        for p in windows
    ]
    networks = defaultdict(list)
    for e in events:
        if (
            e.kind == "pass"
            and e.player_id
            and e.target_id
            and (e.outcome is True or e.qualifiers.get("observed_teammate_transition"))
        ):
            networks[(e.team, e.player_id, e.target_id)].append(e)
    maps = defaultdict(list)
    for e in events:
        maps[e.kind].append(
            {
                "event_id": e.event_id,
                "team": e.team,
                "player_id": e.player_id,
                "timestamp_ms": e.timestamp_ms,
                "position_m": e.position_m,
                "end_position_m": e.end_position_m,
                "spatial_status": "available" if e.position_m is not None else "unavailable",
            }
        )
    for team, metrics in totals.items():
        # Retain measurable camera proxies separately from semantic possession.
        metrics["observed_tracking"] = (
            report.get("teams", {}).get(team, {}) if not annotations else {}
        )
        durations = [m["possession"]["time_s"]["value"] for m in totals.values()]
        statuses = {m["possession"]["time_s"]["status"] for m in totals.values()}
        own_time = metrics["possession"]["time_s"]["value"]
        total_time = sum(durations) if all(t is not None for t in durations) else None
        metrics["possession"]["share_pct"] = {
            "value": round(100 * own_time / total_time, 2) if total_time else None,
            "status": (
                "unavailable"
                if not total_time
                else "estimated"
                if "estimated" in statuses
                else "partial"
                if "partial" in statuses
                else "verified"
            ),
            "unit": "percent",
            "denominator": "sum of annotated possession durations",
        }
        tracking = metrics["observed_tracking"]
        if tracking:
            for key, source_key, factor, unit in (
                ("nearest_ball_time_s", "time_nearest_ball_s", 1, "s"),
                ("nearest_ball_share_pct", "possession_share", 100, "percent"),
            ):
                value = tracking.get(source_key)
                metrics["possession"][key] = {
                    "value": round(value * factor, 3) if value is not None else None,
                    "status": "estimated" if value is not None else "unavailable",
                    "unit": unit,
                    "definition": "nearest-player proxy on observed frames, not semantic control",
                    "source_field": f"teams.{team}.{source_key}",
                }
    return {
        "schema_version": "1.0.0",
        "metric_definition_version": "reference-report-v2",
        "resolved_config": config.model_dump(),
        "match": {
            "match_id": annotations.match_id if annotations else match_id,
            "title": report.get("title"),
            "scoreboard": report.get("scoreboard"),
            "roster": report.get("roster"),
            "coverage_start_ms": start,
            "coverage_end_ms": end,
            "coverage": report.get("coverage"),
            "coverage_basis": report["coverage_basis"],
            "source_duration_ms": duration_ms,
        },
        "provenance": {
            "source": source,
            "predictions_sha256": report.get("predictions_sha256"),
            "upstream_thresholds": report.get("thresholds"),
            "annotation_sha256": hashlib.sha256(annotations.model_dump_json().encode()).hexdigest()
            if annotations
            else None,
            "annotation_source": annotations.source if annotations else None,
            "complete_event_types": sorted(complete),
            "event_status_counts": dict(Counter(e.status for e in events)),
        },
        "teams": totals,
        "players": players,
        "periods": bins,
        "passing_network": [
            {
                "team": team,
                "from_player_id": actor,
                "to_player_id": target,
                "completed": len(items) if all(e.outcome is True for e in items) else None,
                "observed_candidates": len(items),
                "status": "verified" if all(e.status == "verified" for e in items) else "estimated",
                "event_ids": [e.event_id for e in items],
            }
            for (team, actor, target), items in networks.items()
        ],
        "event_maps": dict(maps),
        "events": [e.model_dump(mode="json") for e in events],
        "limitations": [
            "Provider index and proprietary xG are not reverse engineered.",
            "Camera visibility is not minutes played or whole-match distance.",
            "Geometric possession transitions do not establish intentional passes.",
            "Ball pitch coordinates are invalid for airborne-ball geometry.",
            "Period boundaries must come from the match clock; defaults use source video.",
        ],
    }


def _window_events(events: list[ReportEvent], period: Period) -> list[ReportEvent]:
    """Split possession durations at period boundaries; point events use onset."""
    selected = []
    for event in events:
        if event.kind == "possession" and event.end_ms is not None:
            start = max(event.timestamp_ms, period.start_ms)
            end = min(event.end_ms, period.end_ms)
            if start < end:
                selected.append(event.model_copy(update={"timestamp_ms": start, "end_ms": end}))
        elif period.start_ms <= event.timestamp_ms < period.end_ms:
            selected.append(event)
    return selected
