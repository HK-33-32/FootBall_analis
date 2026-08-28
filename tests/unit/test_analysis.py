from __future__ import annotations

import time

from football_intelligence.analysis import (
    WEIGHTS,
    Analysis,
    AnalysisConfig,
    Progress,
    estimate_remaining,
)


def test_before_screening_the_estimate_comes_from_the_length_of_the_file():
    progress = Progress(stage="queued", duration_s=5400.0, rate=24.0)
    # a broadcast is roughly 58% football, so not the whole 90 minutes
    assert 60_000 < estimate_remaining(progress) < 80_000


def test_after_screening_the_estimate_uses_the_football_that_is_left():
    progress = Progress(
        stage="perception", playable_s=600.0, chunks_total=10, chunks_done=4,
        analysed_s=240.0, rate=24.0,
    )
    # 360 s of football left at 24x
    assert abs(estimate_remaining(progress) - (360 * 24 + 6)) < 1


def test_the_estimate_moves_inside_a_chunk_not_only_between_them():
    """A chunk is a minute; an estimate that only ticks per chunk is frozen."""
    early = Progress(
        stage="perception", playable_s=600.0, chunks_total=10, chunks_done=0,
        analysed_s=10.0, rate=24.0,
    )
    later = Progress(
        stage="perception", playable_s=600.0, chunks_total=10, chunks_done=0,
        analysed_s=50.0, rate=24.0,
    )
    assert estimate_remaining(later) < estimate_remaining(early) - 900


def test_a_finished_run_has_nothing_left():
    assert estimate_remaining(Progress(stage="done", playable_s=600.0, chunks_total=4)) == 0.0
    assert estimate_remaining(Progress(stage="error", playable_s=600.0, chunks_total=4)) == 0.0


def test_a_measured_rate_replaces_the_default_in_the_estimate():
    slow = Progress(
        stage="perception", playable_s=100.0, chunks_total=2, analysed_s=50.0, rate=50.0
    )
    fast = Progress(
        stage="perception", playable_s=100.0, chunks_total=2, analysed_s=50.0, rate=10.0
    )
    assert estimate_remaining(slow) > estimate_remaining(fast) * 4


def test_the_stage_weights_cover_the_whole_job():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_progress_moves_forward_stage_by_stage(tmp_path):
    analysis = Analysis(tmp_path / "match.mp4", "Матч", AnalysisConfig(reports_dir=tmp_path))
    seen = []
    for stage in ("screen", "perception", "refine", "report"):
        analysis._step(stage, 1.0)
        seen.append(analysis.progress.fraction)
    assert seen == sorted(seen)
    assert abs(seen[-1] - 1.0) < 1e-9


def test_a_failure_is_reported_rather_than_raised(tmp_path):
    analysis = Analysis(tmp_path / "missing.mp4", "Нет файла", AnalysisConfig(reports_dir=tmp_path))
    analysis.run()
    assert analysis.progress.stage == "error"
    assert analysis.error
    assert analysis.as_dict()["error"]


def test_a_cancelled_run_says_so_rather_than_erroring(tmp_path):
    analysis = Analysis(tmp_path / "missing.mp4", "Отмена", AnalysisConfig(reports_dir=tmp_path))
    analysis.cancel()
    analysis.run()
    assert analysis.progress.stage == "canceled"


def test_the_payload_carries_everything_the_bar_needs(tmp_path):
    analysis = Analysis(tmp_path / "m.mp4", "Матч", AnalysisConfig(reports_dir=tmp_path))
    analysis.progress.duration_s = 100.0
    payload = analysis.as_dict()
    for key in ("stage", "stage_title", "fraction", "eta_s", "elapsed_s", "chunks_total"):
        assert key in payload
    assert payload["stage_title"] == "В очереди"
    assert isinstance(payload["eta_s"], int)
    assert payload["elapsed_s"] >= 0
    assert time.time() >= analysis.progress.started_at


def test_the_bar_never_retreats_when_the_backend_restarts_its_own():
    """The backend runs stages in sequence and resets its progress for each."""
    progress = Progress(stage="perception", playable_s=100.0, chunks_total=1)
    for reported in (0.2, 0.9, 0.3, 0.6, 0.95):
        progress.analysed_s = max(progress.analysed_s, reported * 100.0)
    assert progress.analysed_s == 95.0


def test_the_length_of_a_missing_file_is_zero_not_an_exception(tmp_path):
    from football_intelligence.analysis import video_duration_s

    assert video_duration_s(tmp_path / "nothing.mp4") == 0.0
