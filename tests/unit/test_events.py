from football_intelligence.cache import InferenceCache
from football_intelligence.domain import CandidateScore, EventType
from football_intelligence.events import (
    ActiveSemanticEngine,
    ReanalysisConfig,
    legacy_event_candidates,
)
from football_intelligence.vlm.backend import SemanticRequest, SemanticVLMBackend, VLMOutput


class FakeSampler:
    def sample(self, video_path, start_ms, end_ms, fps, max_frames):
        return [start_ms, end_ms], ["data:image/jpeg;base64,AA==", "data:image/jpeg;base64,AA=="]


class SequenceBackend(SemanticVLMBackend):
    model_name = "test-vlm"
    model_revision = "fixture-sha"
    prompt_version = "fixture-v1"

    def __init__(self):
        self.calls = 0

    def classify(self, request: SemanticRequest) -> VLMOutput:
        self.calls += 1
        if self.calls == 1:
            return VLMOutput(
                primary_event=EventType.INTENTIONAL_PASS,
                confidence=0.55,
                alternatives=[CandidateScore(label=EventType.CLEARANCE, confidence=0.45)],
            )
        return VLMOutput(primary_event=EventType.CLEARANCE, confidence=0.88)


def test_uncertain_first_pass_triggers_higher_quality_second_pass(tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"test-only")
    backend = SequenceBackend()
    engine = ActiveSemanticEngine(backend, config=ReanalysisConfig(), sampler=FakeSampler())
    result = engine.analyse_candidate("m1", "c1", source, 5000, "pass", {}, {})
    assert backend.calls == 2
    assert result.analysis_pass == 2
    assert result.primary_event == EventType.CLEARANCE
    assert "low_event_confidence" in result.trigger_reasons


def test_identical_inference_is_reused_from_content_cache(tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"test-only")
    backend = SequenceBackend()
    engine = ActiveSemanticEngine(
        backend,
        config=ReanalysisConfig(),
        sampler=FakeSampler(),
        cache=InferenceCache(tmp_path / "cache"),
    )
    first = engine.analyse_candidate("m1", "c1", source, 5000, "pass", {}, {})
    calls_after_first = backend.calls
    second = engine.analyse_candidate("m1", "c1", source, 5000, "pass", {}, {})
    assert backend.calls == calls_after_first
    assert second.primary_event == first.primary_event


def test_legacy_pass_is_only_a_candidate_hint():
    candidates = legacy_event_candidates(
        {
            "events": [
                {
                    "event_id": "x",
                    "type": "pass",
                    "clock": {"video_time_s": 12.3},
                    "confidence": 0.8,
                }
            ]
        }
    )
    assert candidates == [
        {
            "candidate_id": "x",
            "centre_ms": 12300,
            "candidate_hint": "pass",
            "geometry": {
                "start_xy": None,
                "end_xy": None,
                "attributes": {},
                "legacy_confidence": 0.8,
            },
            "player_context": {"actor_tracklet_id": None, "target_tracklet_id": None, "team": None},
            "important": False,
        }
    ]
