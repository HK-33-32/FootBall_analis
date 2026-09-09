import json

import pytest
from pydantic import ValidationError

from football_intelligence.reporting import (
    Period,
    ReportAnnotations,
    ReportConfig,
    ReportEvent,
    detailed_report,
    report_with_duration,
)


def test_known_duration_counts_missing_leading_and_trailing_frames():
    report = {"fps": 25, "first_frame": 26, "last_frame": 50,
              "frames_with_game_state": 25, "coverage": 1, "teams": {}, "players": []}
    result = detailed_report(report, duration_ms=4000)
    assert result["match"]["coverage"] == .25
    assert result["match"]["coverage_start_ms"] == 0
    assert result["match"]["coverage_end_ms"] == 4000
    assert result["match"]["coverage_basis"] == "source_duration"


def test_unknown_duration_does_not_claim_full_clip_coverage():
    report = {"fps": 25, "first_frame": 26, "last_frame": 50,
              "frames_with_game_state": 25, "coverage": 1, "teams": {}, "players": []}
    result = detailed_report(report)
    assert result["match"]["coverage_basis"] == "observed_prediction_span"
    assert result["match"]["source_duration_ms"] is None


def test_duration_corrects_player_coverage_without_mutating_input():
    report = {"fps": 25, "last_frame": 50, "frames_with_game_state": 25, "coverage": 1,
              "players": [{"samples_used": 20, "coverage": .8}]}
    result = report_with_duration(report, 4000)
    assert result["players"][0]["coverage"] == .2
    assert report["players"][0]["coverage"] == .8


@pytest.mark.parametrize("duration", [0, -1, 1000, float("nan")])
def test_invalid_duration_is_not_silently_used(duration):
    with pytest.raises(ValueError):
        report_with_duration({"fps": 25, "last_frame": 50}, duration)


def event(event_id="e1", **kwargs):
    return ReportEvent(
        event_id=event_id,
        source="test-only-reviewed-fixture",
        evidence_ids=["test://clip#t=1"],
        status="verified",
        **(
            {"kind": "pass", "timestamp_ms": 1000, "team": "A", "player_id": "a1", "outcome": True}
            | kwargs
        ),
    )


def annotated(events, complete=(), **kwargs):
    return detailed_report(
        {"fps": 25, "teams": {"left": {}, "right": {}}, "players": []},
        annotations=ReportAnnotations(
            match_id="fixture",
            source="test-fixture",
            teams=["A", "B"],
            events=events,
            complete_event_types=list(complete),
            coverage_end_ms=60_000,
            **kwargs,
        ),
    )


def test_complete_zero_is_not_missing_coverage():
    missing = annotated([])["teams"]["A"]["shots"]["total"]
    assert missing["value"] is None
    zero = annotated([], ["shot"])["teams"]["A"]["shots"]["total"]
    assert zero["value"] == 0 and zero["status"] == "verified"


def test_unattributed_events_prevent_false_verified_team_zeroes():
    report = annotated([event(kind="shot", team=None)], ["shot"])
    assert report["teams"]["A"]["shots"]["total"]["value"] is None
    assert report["teams"]["B"]["shots"]["total"]["value"] is None
    assert report["events"][0]["team"] is None


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize(
    "start,end,progressive", [(-40, -10, True), (-5, 8, False), (-10, 10, True), (8, 18, True)]
)
def test_progressive_pass_definition_changes_with_pitch_zone_and_direction(
    sign, start, end, progressive
):
    e = event(position_m=(start * sign, 0), end_position_m=(end * sign, 0), attacking_sign=sign)
    stats = annotated([e], ["pass"])["teams"]["A"]["passes"]
    assert stats["progressive"]["value"] == int(progressive)
    assert stats["forward"]["value"] == 1


def test_success_denominator_includes_unsuccessful_passes():
    report = annotated([event(), event("e2", outcome=False)], ["pass"])
    metric = report["teams"]["A"]["passes"]["total"]
    assert metric["successful"] == 1 and metric["success_pct"] == 50
    assert set(report["teams"]) == {"A", "B"}  # no accidental left/right aliases


def test_unknown_pass_outcome_does_not_become_failure_or_success():
    metric = annotated([event(outcome=None)])["teams"]["A"]["passes"]["total"]
    assert metric["value"] == 1
    assert metric["successful"] is None and metric["success_pct"] is None
    assert metric["status"] == "partial"


def test_xg_requires_model_provenance_for_every_shot():
    metric = annotated([event(kind="shot", qualifiers={"xg": 0.2})])["teams"]["A"]["shots"]
    assert metric["xg"]["value"] is None
    metric = annotated(
        [event(kind="shot", qualifiers={"xg": 0.2, "xg_model": "fixture-v1"})], ["shot"]
    )["teams"]["A"]["shots"]
    assert metric["xg"]["value"] == 0.2


def test_missing_qualifiers_are_not_negative_evidence():
    stats = annotated([event(kind="shot", qualifiers={"on_target": "false"})])["teams"]["A"]
    assert stats["shots"]["on_target"]["value"] is None
    assert stats["shots"]["blocked"]["value"] is None


def test_ppda_requires_complete_spatial_coverage_and_handles_zero_denominator():
    events = [
        event(team="B", attacking_sign=-1, position_m=(20, 0)),
        event("e2", kind="tackle", attacking_sign=1, position_m=(20, 0)),
    ]
    required = ["pass", "tackle", "interception", "duel", "foul"]
    assert annotated(events)["teams"]["A"]["pressing"]["ppda"]["value"] is None
    report = annotated(events, required)
    metric = report["teams"]["A"]["pressing"]["ppda"]
    assert metric["value"] == 1 and metric["status"] == "verified"
    assert report["teams"]["B"]["pressing"]["ppda"]["value"] is None


def test_possession_is_split_at_period_boundaries():
    periods = [
        Period(name="first", start_ms=0, end_ms=20_000),
        Period(name="second", start_ms=20_000, end_ms=60_000),
    ]
    events = [
        event(kind="possession", timestamp_ms=10_000, end_ms=30_000),
        event("e2", timestamp_ms=20_000),
    ]
    report = annotated(events, ["possession", "pass"], periods=periods)
    first, second = [p["teams"]["A"] for p in report["periods"]]
    assert first["possession"]["time_s"]["value"] == 10
    assert second["possession"]["time_s"]["value"] == 10
    assert first["passes"]["total"]["value"] == 0
    assert second["passes"]["total"]["value"] == 1
    assert report["teams"]["A"]["possession"]["share_pct"]["value"] == 100


def test_maps_networks_and_recipients_resolve_to_stable_evidence():
    report = annotated([event(target_id="a2", position_m=(0, 0), end_position_m=(10, 1))], ["pass"])
    assert report["passing_network"][0]["completed"] == 1
    assert report["passing_network"][0]["event_ids"] == ["e1"]
    assert report["event_maps"]["pass"][0]["position_m"] == (0, 0)
    assert {p["player_id"] for p in report["players"]} == {"a1", "a2"}
    assert all(p["identity_scope"] == "annotation_provider" for p in report["players"])
    json.dumps(report, allow_nan=False)


def test_geometric_pass_is_not_100_percent_semantic_accuracy():
    base = {
        "fps": 25,
        "first_frame": 1,
        "last_frame": 50,
        "video_source": "video.mp4",
        "teams": {"left": {}},
        "events": [
            {
                "kind": "pass",
                "frame": 1,
                "team": "left",
                "player": 1,
                "to": 2,
                "flight_s": 0.4,
                "progressive": True,
            }
        ],
    }
    predictions = [
        {
            "frame": 1,
            "track_id": 1,
            "attributes": {"role": "player"},
            "bbox_pitch": {"x_bottom_middle": 0, "y_bottom_middle": 0},
        }
    ]
    report = detailed_report(base, predictions=predictions)
    assert report["events"][0]["timestamp_ms"] == 0
    assert report["events"][0]["position_m"] == [0, 0]
    assert report["teams"]["left"]["passes"]["total"]["status"] == "estimated"
    assert report["teams"]["left"]["passes"]["total"]["success_pct"] is None
    assert report["teams"]["left"]["passes"]["progressive"]["value"] is None
    assert report["passing_network"][0]["completed"] is None
    assert report["passing_network"][0]["observed_candidates"] == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"end_ms": 0},
        {"position_m": (float("nan"), 0)},
        {"qualifiers": {"distance_m": -1}},
        {"qualifiers": {"xg": 1.5}},
        {"evidence_ids": []},
    ],
)
def test_invalid_event_is_rejected(overrides):
    payload = event().model_dump() | overrides
    with pytest.raises(ValidationError):
        ReportEvent.model_validate(payload)


def test_duplicate_ids_outside_coverage_and_overlap_are_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        annotated([event(), event()])
    with pytest.raises(ValidationError, match="outside"):
        annotated([event(end_ms=61_000)])
    with pytest.raises(ValidationError, match="overlap"):
        annotated(
            [
                event(kind="possession", end_ms=4000),
                event("e2", kind="possession", timestamp_ms=2000, end_ms=5000),
            ]
        )


def test_configuration_is_validated_and_saved():
    with pytest.raises(ValidationError):
        ReportConfig(short_pass_max_m=41)
    config = ReportConfig(interval_ms=10_000)
    report = detailed_report({"fps": 25, "first_frame": 1, "last_frame": 1000}, config=config)
    assert report["resolved_config"] == config.model_dump()
    assert len(report["periods"]) == 4
