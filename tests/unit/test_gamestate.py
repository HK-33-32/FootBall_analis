from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from football_intelligence.gamestate import (
    JerseyTally,
    RefinementConfig,
    TrackSummary,
    _cluster_kits,
    assign_teams,
    demote_stranded_goalkeepers,
    link_tracklets,
    load_reid_embeddings,
    refine_predictions,
    resolve_jerseys,
    summarise_tracks,
    torso_descriptor,
)


class Tracklet:
    """Same class name the perception backend pickles its tracklet dump with."""

    def __init__(self, times, bboxes, features):
        self.times, self.bboxes, self.features = times, bboxes, features


GREEN_BGR = (60, 140, 60)
WHITE_KIT = (235, 235, 235)
DARK_KIT = (40, 40, 150)


def _frame(boxes: dict[tuple[int, int], tuple[int, int, int]]) -> np.ndarray:
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    frame[:, :] = GREEN_BGR
    for (x, y), colour in boxes.items():
        frame[y : y + 80, x : x + 40] = colour
    return frame


def _detection(
    frame: int,
    track_id: int,
    x: float,
    pitch_x: float,
    *,
    role: str = "player",
    team: str = "left",
    jersey: str | None = None,
    pitch_y: float = 0.0,
    height: float = 80.0,
) -> dict:
    return {
        "frame": frame,
        "track_id": track_id,
        "bbox_image": {"x": x, "y": 10.0, "w": 40.0, "h": height},
        "bbox_pitch": {"x_bottom_middle": pitch_x, "y_bottom_middle": pitch_y},
        "attributes": {"role": role, "team": team, "jersey": jersey, "name": None},
    }


def test_torso_descriptor_ignores_grass_and_small_boxes():
    config = RefinementConfig()
    frame = _frame({(100, 10): WHITE_KIT})
    bright = torso_descriptor(frame, {"x": 100, "y": 10, "w": 40, "h": 80}, config)
    assert bright is not None
    assert bright[0] > 200  # lightness of a white kit

    dark = torso_descriptor(
        _frame({(100, 10): DARK_KIT}), {"x": 100, "y": 10, "w": 40, "h": 80}, config
    )
    assert dark is not None
    assert dark[0] < bright[0]

    assert torso_descriptor(frame, {"x": 100, "y": 10, "w": 6, "h": 12}, config) is None


def test_torso_descriptor_rejects_a_pure_grass_box():
    config = RefinementConfig()
    frame = _frame({})
    assert torso_descriptor(frame, {"x": 100, "y": 10, "w": 40, "h": 80}, config) is None


def _two_team_predictions() -> tuple[list[dict], dict[int, np.ndarray]]:
    """Four white tracks on the left, four dark tracks on the right, plus a keeper."""
    predictions: list[dict] = []
    layout = {
        1: (WHITE_KIT, -30.0),
        2: (WHITE_KIT, -22.0),
        3: (WHITE_KIT, -14.0),
        4: (WHITE_KIT, -6.0),
        5: (DARK_KIT, 6.0),
        6: (DARK_KIT, 14.0),
        7: (DARK_KIT, 22.0),
        8: (DARK_KIT, 30.0),
    }
    frames: dict[int, np.ndarray] = {}
    for frame_number in range(1, 11):
        boxes = {}
        for position, (track_id, (colour, pitch_x)) in enumerate(layout.items()):
            x = 20 + position * 45
            boxes[(x, 10)] = colour
            predictions.append(
                _detection(frame_number, track_id, x, pitch_x, team="left", pitch_y=position)
            )
        predictions.append(
            _detection(frame_number, 99, 380, 48.0, role="goalkeeper", team="left", pitch_y=0.0)
        )
        frames[frame_number] = _frame(boxes)
    return predictions, frames


def test_assign_teams_recovers_kit_clusters_and_pitch_sides():
    predictions, frames = _two_team_predictions()
    config = RefinementConfig(frame_stride=1)
    summaries = summarise_tracks(predictions, frames.get, config)
    report = assign_teams(summaries, predictions, config)

    assert report["status"] == "ok"
    assert report["goalkeeper_side"] == "right"
    assert report["side_rule"] == "offside-line-vote"
    assert report["sides_agree"] is True
    assert {summaries[track].team for track in (1, 2, 3, 4)} == {"left"}
    assert {summaries[track].team for track in (5, 6, 7, 8)} == {"right"}
    assert summaries[99].team == "right"


def test_kits_cluster_by_colour_not_by_how_brightly_lit_the_player_is():
    """Two kits, each seen once in full light and once in shade."""
    lit_red, shaded_red = (60, 60, 200), (24, 24, 80)
    lit_blue, shaded_blue = (200, 60, 60), (80, 24, 24)
    layout = {1: lit_red, 2: shaded_red, 3: lit_blue, 4: shaded_blue}

    predictions: list[dict] = []
    frames: dict[int, np.ndarray] = {}
    for frame_number in range(1, 11):
        boxes = {}
        for position, (track_id, colour) in enumerate(layout.items()):
            x = 20 + position * 60
            boxes[(x, 10)] = colour
            predictions.append(_detection(frame_number, track_id, x, -20.0 + 10 * position))
        frames[frame_number] = _frame(boxes)

    def clusters(lightness_weight: float) -> set[frozenset[int]]:
        config = RefinementConfig(frame_stride=1, lightness_weight=lightness_weight)
        summaries = summarise_tracks(predictions, frames.get, config)
        assign_teams(summaries, predictions, config)
        grouped: dict[int, set[int]] = {0: set(), 1: set()}
        for track_id, summary in summaries.items():
            grouped[summary.team_cluster].add(track_id)
        return {frozenset(members) for members in grouped.values()}

    assert clusters(0.0) == {frozenset({1, 2}), frozenset({3, 4})}
    assert clusters(1.0) != {frozenset({1, 2}), frozenset({3, 4})}


def test_a_lopsided_chroma_split_is_retried_with_lightness():
    """Both teams are on the pitch, so a 5-1 split is not a split by kit."""
    weights = np.array([9.0, 9.0, 9.0, 9.0, 9.0, 9.0])
    # Chroma alone separates one player from the rest; lightness separates 3/3.
    descriptors = np.array(
        [
            [10.0, 1.0, 1.0],
            [90.0, 1.2, 0.8],
            [12.0, 0.9, 1.1],
            [88.0, 40.0, 40.0],
            [11.0, 1.1, 0.9],
            [91.0, 1.0, 1.0],
        ]
    )
    config = RefinementConfig()
    labels, _, scale, _ = _cluster_kits(descriptors, weights, config)
    assert scale[0] == 1.0  # the retry was kept
    assert sorted(np.bincount(labels, minlength=2).tolist()) == [3, 3]

    patient = RefinementConfig(max_kit_mass_share=1.0)
    labels, _, scale, _ = _cluster_kits(descriptors, weights, patient)
    assert scale[0] == 0.0  # without the balance rule the lopsided split stands
    assert sorted(np.bincount(labels, minlength=2).tolist()) == [1, 5]


def test_a_balanced_chroma_split_is_left_alone():
    weights = np.array([9.0, 9.0, 9.0, 9.0])
    descriptors = np.array(
        [[10.0, 40.0, 5.0], [80.0, 41.0, 6.0], [12.0, 5.0, 40.0], [78.0, 6.0, 41.0]]
    )
    _, _, scale, _ = _cluster_kits(descriptors, weights, RefinementConfig())
    assert scale[0] == 0.0


def test_assign_teams_reports_insufficient_evidence_without_crops():
    predictions = [_detection(1, 1, 10.0, 0.0), _detection(2, 1, 10.0, 0.0)]
    config = RefinementConfig(frame_stride=1)
    summaries = summarise_tracks(predictions, lambda _: None, config)
    assert assign_teams(summaries, predictions, config)["status"] == "insufficient-evidence"


def _summary(track_id: int, start: int, end: int, first, last, **kwargs) -> TrackSummary:
    return TrackSummary(
        track_id=track_id,
        role=kwargs.pop("role", "player"),
        team=kwargs.pop("team", "left"),
        start_frame=start,
        end_frame=end,
        first_point=first,
        last_point=last,
        detections=end - start + 1,
        **kwargs,
    )


def test_link_tracklets_joins_a_stationary_gap_and_refuses_a_moved_player():
    config = RefinementConfig(link_max_gap_frames=300, link_max_distance_m=5.0)
    summaries = {
        1: _summary(1, 1, 100, (0.0, 0.0), (10.0, 5.0)),
        2: _summary(2, 200, 300, (10.4, 5.2), (12.0, 6.0)),
        3: _summary(3, 400, 500, (40.0, 20.0), (41.0, 21.0)),
    }
    remap, report = link_tracklets(summaries, config)
    assert report["links"] == 1
    assert remap[2] == 1
    assert remap[3] == 3


def test_link_tracklets_refuses_overlapping_or_cross_team_pairs():
    config = RefinementConfig()
    overlapping = {
        1: _summary(1, 1, 300, (0.0, 0.0), (1.0, 0.0)),
        2: _summary(2, 200, 400, (1.1, 0.0), (2.0, 0.0)),
    }
    assert link_tracklets(overlapping, config)[1]["links"] == 0

    cross_team = {
        1: _summary(1, 1, 100, (0.0, 0.0), (1.0, 0.0), team="left"),
        2: _summary(2, 150, 300, (1.1, 0.0), (2.0, 0.0), team="right"),
    }
    assert link_tracklets(cross_team, config)[1]["links"] == 0


def test_resolve_jerseys_prefers_the_larger_crop_within_one_identity():
    summaries = {
        1: _summary(1, 1, 100, (0.0, 0.0), (0.0, 0.0), median_height=70.0),
        2: _summary(2, 200, 300, (0.0, 0.0), (0.0, 0.0), median_height=170.0),
    }
    summaries[1].jersey_votes.update({"2": 400})
    summaries[2].jersey_votes.update({"22": 100})

    resolve_jerseys(summaries, RefinementConfig(), {1: 1, 2: 1})
    assert summaries[1].jersey == "22"
    assert summaries[2].jersey == "22"

    resolve_jerseys(summaries, RefinementConfig(), None)
    assert summaries[1].jersey == "2"
    assert summaries[2].jersey == "22"


def test_resolve_jerseys_abstains_below_the_vote_floor():
    summaries = {1: _summary(1, 1, 100, (0.0, 0.0), (0.0, 0.0), median_height=120.0)}
    summaries[1].jersey_votes.update({"7": 1})
    resolve_jerseys(summaries, RefinementConfig(jersey_min_votes=2), None)
    assert summaries[1].jersey is None


def test_refine_predictions_rewrites_teams_and_propagates_jerseys():
    predictions, frames = _two_team_predictions()
    for detection in predictions:
        if detection["track_id"] == 5 and detection["frame"] > 8:
            detection["attributes"]["jersey"] = "9"
    config = RefinementConfig(frame_stride=1, jersey_fill=True, link_tracklets=True)
    refined, report = refine_predictions(predictions, frames.get, config)

    assert report["team"]["status"] == "ok"
    assert report["team_labels_changed"] == 50  # four dark tracks plus the keeper, ten frames
    teams = {
        detection["track_id"]: detection["attributes"]["team"]
        for detection in refined
        if detection["attributes"]["role"] == "player"
    }
    assert teams[1] == "left" and teams[5] == "right"
    jerseys = {
        detection["frame"]: detection["attributes"]["jersey"]
        for detection in refined
        if detection["track_id"] == 5
    }
    assert set(jerseys.values()) == {"9"}  # filled backwards over the whole tracklet
    assert all(
        detection["attributes"]["jersey"] is None
        for detection in refined
        if detection["track_id"] == 1
    )


def test_jersey_fill_is_separate_from_propagation_along_a_link():
    """A tracklet's own gaps and a linked sibling's gaps are different bets."""
    predictions, frames = _two_team_predictions()
    for detection in predictions:
        if detection["track_id"] == 5 and detection["frame"] > 8:
            detection["attributes"]["jersey"] = "9"

    def jerseys(config: RefinementConfig) -> set:
        refined, _ = refine_predictions(predictions, frames.get, config)
        return {
            detection["attributes"]["jersey"] for detection in refined if detection["track_id"] == 5
        }

    base = RefinementConfig(frame_stride=1, link_tracklets=True)
    assert jerseys(base) == {None, "9"}  # the reader's own abstentions are kept
    assert jerseys(RefinementConfig(frame_stride=1, link_tracklets=True, jersey_fill=True)) == {"9"}


def test_load_reid_embeddings_matches_detections_by_box(tmp_path):
    """The backend pickles instances of its own ``Tracklet`` class."""
    import pickle

    dump = {
        7: Tracklet(
            [1, 2],
            [[20.0, 10.0, 40.0, 80.0], [65.0, 10.0, 40.0, 80.0]],
            [np.array([1.0, 0.0]), np.array([0.0, 2.0])],
        )
    }
    path = tmp_path / "tracklets.pkl"
    path.write_bytes(pickle.dumps(dump))

    predictions = [
        _detection(1, 3, 20.0, -30.0),
        _detection(2, 4, 65.0, -22.0),
        _detection(2, 5, 900.0, 5.0),
    ]
    embeddings = load_reid_embeddings(str(path), predictions)
    assert set(embeddings) == {3, 4}
    assert np.allclose(embeddings[3], [1.0, 0.0])
    assert np.allclose(embeddings[4], [0.0, 1.0])


def test_link_tracklets_uses_embeddings_to_separate_look_alikes():
    config = RefinementConfig(reid_min_cosine=0.9, reid_max_distance_m=30.0)
    summaries = {
        1: _summary(1, 1, 100, (0.0, 0.0), (10.0, 0.0)),
        2: _summary(2, 200, 300, (25.0, 0.0), (26.0, 0.0)),
    }
    far = np.array([1.0, 0.0])
    near = np.array([0.995, 0.0999])
    assert link_tracklets(summaries, config, {1: far, 2: near})[1]["links"] == 1
    assert link_tracklets(summaries, config, {1: far, 2: np.array([0.0, 1.0])})[1]["links"] == 0
    # 25 m apart is outside the geometry-only gate, so no link without embeddings
    assert link_tracklets(summaries, config, None)[1]["links"] == 0


def test_refine_predictions_leaves_non_player_rows_alone():
    ball = {
        "frame": 1,
        "track_id": 500,
        "bbox_image": {"x": 1.0, "y": 1.0, "w": 4.0, "h": 4.0},
        "bbox_pitch": {"x_bottom_middle": 0.0, "y_bottom_middle": 0.0},
        "attributes": {"role": "ball", "jersey": None, "team": None},
    }
    predictions, frames = _two_team_predictions()
    refined, _ = refine_predictions(
        [*predictions, ball], frames.get, RefinementConfig(frame_stride=1)
    )
    assert refined[-1]["attributes"] == ball["attributes"]
    assert refined[-1]["track_id"] == 500


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_offside_rank_is_configurable_without_flipping_the_answer(rank):
    predictions, frames = _two_team_predictions()
    config = RefinementConfig(frame_stride=1, offside_rank=rank)
    summaries = summarise_tracks(predictions, frames.get, config)
    report = assign_teams(summaries, predictions, config)
    assert report["resolved_sides"] == report["fallback_sides"]


def _read(votes: dict[str, int], questions: int) -> JerseyTally:
    return JerseyTally(votes=Counter(votes), questions=questions)


def test_identity_reread_overrides_a_confident_but_wrong_tracklet_read():
    summaries = {
        1: _summary(1, 1, 100, (0.0, 0.0), (0.0, 0.0), median_height=170.0),
    }
    summaries[1].jersey_votes.update({"2": 300})
    resolve_jerseys(summaries, RefinementConfig(), None, {1: _read({"22": 5}, 6)})
    assert summaries[1].jersey == "22"


def test_a_reader_that_contradicts_itself_is_ignored_rather_than_blended():
    summaries = {1: _summary(1, 1, 100, (0.0, 0.0), (0.0, 0.0), median_height=60.0)}
    summaries[1].jersey_votes.update({"9": 300})
    scattered = {1: _read({"10": 1, "7": 1, "11": 1}, 6)}

    resolve_jerseys(summaries, RefinementConfig(), None, scattered)
    assert summaries[1].jersey == "9"

    # blended in, three mutually contradictory guesses do not merely mislead:
    # they split the mass until nothing clears the decision threshold
    resolve_jerseys(summaries, RefinementConfig(identity_read_min_consensus=0.0), None, scattered)
    assert summaries[1].jersey is None


def test_two_teammates_on_the_pitch_together_cannot_wear_the_same_number():
    summaries = {
        1: _summary(1, 1, 200, (0.0, 0.0), (0.0, 0.0), median_height=150.0, team="left"),
        2: _summary(2, 50, 250, (0.0, 0.0), (0.0, 0.0), median_height=150.0, team="left"),
    }
    summaries[2].jersey_votes.update({"4": 100})
    reads = {1: _read({"10": 6}, 6), 2: _read({"10": 4, "4": 1}, 6)}

    resolve_jerseys(summaries, RefinementConfig(), None, reads)
    assert summaries[1].jersey == "10"  # the stronger claim keeps the number
    assert summaries[2].jersey == "4"  # the weaker one falls back

    kept = resolve_jerseys(
        summaries, RefinementConfig(enforce_number_exclusivity=False), None, reads
    )
    assert kept["number_conflicts_resolved"] == 0
    assert summaries[1].jersey == summaries[2].jersey == "10"


def test_number_exclusivity_spares_identities_that_never_share_the_pitch():
    """Two disjoint spans may simply be one player that linking failed to join."""
    summaries = {
        1: _summary(1, 1, 100, (0.0, 0.0), (0.0, 0.0), median_height=150.0, team="left"),
        2: _summary(2, 300, 400, (0.0, 0.0), (0.0, 0.0), median_height=150.0, team="left"),
    }
    reads = {1: _read({"10": 6}, 6), 2: _read({"10": 4}, 6)}
    report = resolve_jerseys(summaries, RefinementConfig(), None, reads)
    assert report["number_conflicts_resolved"] == 0
    assert summaries[1].jersey == summaries[2].jersey == "10"


def test_number_exclusivity_does_not_cross_teams():
    summaries = {
        1: _summary(1, 1, 200, (0.0, 0.0), (0.0, 0.0), median_height=150.0, team="left"),
        2: _summary(2, 1, 200, (0.0, 0.0), (0.0, 0.0), median_height=150.0, team="right"),
    }
    reads = {1: _read({"8": 6}, 6), 2: _read({"8": 4}, 6)}
    resolve_jerseys(summaries, RefinementConfig(), None, reads)
    assert summaries[1].jersey == summaries[2].jersey == "8"


def test_jersey_tally_consensus_counts_refusals():
    assert _read({"7": 3}, 6).consensus == 0.5
    assert _read({"7": 3}, 3).consensus == 1.0
    assert JerseyTally(votes=Counter(), questions=6).consensus == 0.0


def test_a_goalkeeper_standing_in_midfield_is_demoted_to_player():
    summaries = {
        1: _summary(1, 1, 100, (0.0, 0.0), (0.0, 0.0), role="goalkeeper"),
        2: _summary(2, 1, 100, (0.0, 0.0), (0.0, 0.0), role="goalkeeper"),
    }
    summaries[1].pitch_x = -4.4
    summaries[2].pitch_x = -46.7
    assert demote_stranded_goalkeepers(summaries, RefinementConfig()) == [1]
    assert summaries[1].role == "player"
    assert summaries[2].role == "goalkeeper"


def test_role_reconciliation_can_be_turned_off():
    summaries = {1: _summary(1, 1, 100, (0.0, 0.0), (0.0, 0.0), role="goalkeeper")}
    summaries[1].pitch_x = 0.0
    predictions = [_detection(1, 1, 10.0, 0.0, role="goalkeeper")]
    assign_teams(summaries, predictions, RefinementConfig(reconcile_roles=False))
    assert summaries[1].role == "goalkeeper"


def test_refine_predictions_writes_the_reconciled_role_back():
    predictions, frames = _two_team_predictions()
    for detection in predictions:
        if detection["track_id"] == 99:
            detection["bbox_pitch"]["x_bottom_middle"] = 2.0  # keeper stranded at halfway
    refined, report = refine_predictions(predictions, frames.get, RefinementConfig(frame_stride=1))
    assert report["team"]["goalkeepers_demoted"] == [99]
    assert report["role_labels_changed"] == 10
    assert {
        detection["attributes"]["role"] for detection in refined if detection["track_id"] == 99
    } == {"player"}
