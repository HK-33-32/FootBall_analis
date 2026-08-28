from __future__ import annotations

import json

import pytest

from football_intelligence.viewer import library


def _write_match(root, name, *, title="Матч", video="clip_web.mp4", players=3):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / library.REPORT_NAME).write_text(
        json.dumps(
            {
                "title": title,
                "clip_frames": 750,
                "frames_with_game_state": 250,
                "players": [{"track": i} for i in range(players)],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if video:
        (directory / video).write_bytes(b"not really a video")
    return directory


def test_a_directory_of_matches_is_listed_newest_first(tmp_path):
    _write_match(tmp_path, "older", title="Первый")
    later = _write_match(tmp_path, "newer", title="Второй")
    # touch the second one so its report is unambiguously the newer file
    (later / library.REPORT_NAME).touch()

    found = library.discover(tmp_path)
    assert [item.id for item in found] == ["newer", "older"]
    assert found[0].title == "Второй"
    assert found[0].players == 3
    assert found[0].covered == 250
    assert found[0].video is not None


def test_a_missing_directory_is_empty_not_an_error(tmp_path):
    assert library.discover(tmp_path / "nothing-here") == []


def test_a_half_written_report_is_skipped_rather_than_breaking_the_listing(tmp_path):
    _write_match(tmp_path, "good")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / library.REPORT_NAME).write_text("{not json", encoding="utf-8")

    assert [item.id for item in library.discover(tmp_path)] == ["good"]


def test_the_annotated_render_is_not_mistaken_for_the_match(tmp_path):
    directory = _write_match(tmp_path, "clip", video=None)
    (directory / "annotated_comparison.mp4").write_bytes(b"diagnostic")
    (directory / "raw.mp4").write_bytes(b"the match")

    assert library.find(tmp_path, "clip").video.name == "raw.mp4"


def test_a_match_without_a_clip_still_lists(tmp_path):
    _write_match(tmp_path, "telemetry-only", video=None)
    found = library.find(tmp_path, "telemetry-only")
    assert found is not None and found.video is None
    assert found.as_dict()["has_video"] is False


@pytest.mark.parametrize("identifier", ["../secrets", "a/b", "", ".hidden", "x" * 200])
def test_an_id_that_is_not_a_plain_name_is_refused(tmp_path, identifier):
    _write_match(tmp_path, "real")
    assert library.find(tmp_path, identifier) is None


def test_rendering_puts_the_report_and_the_video_url_into_the_page():
    template = "<title>Pitch Telemetry</title><video src='__VIDEO_SRC__'>__MATCH_DATA__"
    html = library.render({"title": "Матч"}, "/api/v1/reports/x/video", template, title="Матч")
    assert "__MATCH_DATA__" not in html and "__VIDEO_SRC__" not in html
    assert "/api/v1/reports/x/video" in html
    assert "<title>Матч</title>" in html
    assert '{"title":"Матч"}' in html


def test_rendering_cannot_be_closed_early_by_the_payload():
    """A report carrying "</script>" must not end the block it sits in."""
    template = "__MATCH_DATA__"
    html = library.render({"title": "</script><img onerror=alert(1)>"}, "", template)
    assert "</script>" not in html
    assert "<\\/script>" in html


def test_a_template_without_the_placeholder_is_an_error():
    with pytest.raises(ValueError, match="__MATCH_DATA__"):
        library.render({}, "", "<html>nothing here</html>")
