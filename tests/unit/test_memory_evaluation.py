from football_intelligence.domain import EventType, EvidenceRef, SemanticEvent
from football_intelligence.evaluation import semantic_metrics
from football_intelligence.memory import MatchMemory


def event(identifier: str, kind: EventType, confidence: float = 0.8):
    return SemanticEvent(
        event_id=identifier,
        match_id="m1",
        primary_event=kind,
        confidence=confidence,
        start_ms=1000,
        end_ms=2000,
        evidence=[
            EvidenceRef(
                evidence_id="e_" + identifier,
                match_id="m1",
                source_uri="/video.mp4",
                start_ms=1000,
                end_ms=2000,
                timestamps_ms=[1500],
            )
        ],
        model_name="test-model",
        model_revision="sha",
        prompt_version="v1",
    )


def test_match_memory_preserves_resolvable_evidence(tmp_path):
    memory = MatchMemory(tmp_path / "memory.sqlite")
    memory.put_events([event("one", EventType.INTENTIONAL_PASS), event("two", EventType.CLEARANCE)])
    assert len(memory.events("m1")) == 2
    assert memory.evidence("e_one")["source_uri"] == "/video.mp4"
    assert memory.statistics("m1")["counts"] == {"clearance": 1, "intentional_pass": 1}


def test_semantic_metrics_reports_macro_f1_coverage_and_ece():
    metrics = semantic_metrics(
        [
            {"truth": "pass", "prediction": "pass", "confidence": 0.9},
            {"truth": "clearance", "prediction": "pass", "confidence": 0.6},
            {
                "truth": "clearance",
                "prediction": "unknown",
                "confidence": 0.4,
                "insufficient_evidence": True,
            },
        ],
        labels=["pass", "clearance", "unknown"],
    )
    assert metrics["sample_size"] == 3
    assert metrics["coverage"] == 2 / 3
    assert metrics["accuracy"] == 0.5
    assert 0 <= metrics["ece"] <= 1
