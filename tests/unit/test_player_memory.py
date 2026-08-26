from football_intelligence.domain import JerseyEvidence, ViewObservation
from football_intelligence.player_memory import (
    BestViewConfig,
    BestViewSelector,
    aggregate_jersey_posterior,
)


def view(identifier: str, timestamp: int, sharpness: float, embedding: list[float]):
    return ViewObservation(
        observation_id=identifier,
        tracklet_id="t1",
        timestamp_ms=timestamp,
        frame_index=timestamp // 40,
        crop_uri=f"crop://{identifier}",
        bbox_xyxy=(0, 0, 100, 200),
        sharpness=sharpness,
        resolution_score=0.9,
        occlusion=0.05,
        back_visibility=0.9,
        front_visibility=0.1,
        side_visibility=0.2,
        track_confidence=0.95,
        embedding=embedding,
    )


def test_best_view_selector_enforces_temporal_and_visual_diversity():
    selector = BestViewSelector(
        BestViewConfig(top_k=3, min_time_gap_ms=500, max_cosine_similarity=0.95)
    )
    selected = selector.select(
        [
            view("best", 1000, 500, [1, 0]),
            view("too-close", 1200, 490, [0, 1]),
            view("duplicate", 2000, 480, [1, 0.01]),
            view("diverse", 2600, 300, [0, 1]),
        ]
    )
    assert [item.observation_id for item in selected] == ["best", "diverse"]


def test_jersey_posterior_filters_roster_and_deduplicates_observation():
    evidence = [
        JerseyEvidence(
            number=7,
            confidence=0.9,
            observation_id="a",
            model_name="m",
            model_revision="r",
            prompt_version="p",
        ),
        JerseyEvidence(
            number=7,
            confidence=0.7,
            observation_id="a",
            model_name="m",
            model_revision="r",
            prompt_version="p",
        ),
        JerseyEvidence(
            number=7,
            confidence=0.8,
            observation_id="b",
            model_name="m",
            model_revision="r",
            prompt_version="p",
        ),
        JerseyEvidence(
            number=99,
            confidence=0.99,
            observation_id="c",
            model_name="m",
            model_revision="r",
            prompt_version="p",
        ),
    ]
    posterior = aggregate_jersey_posterior(evidence, allowed_numbers={7, 10})
    assert set(posterior) == {7, 10}
    assert posterior[7] > 0.95
    assert abs(sum(posterior.values()) - 1) < 1e-9
