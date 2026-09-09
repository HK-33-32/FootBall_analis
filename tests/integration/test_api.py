from fastapi.testclient import TestClient

from football_intelligence.api.app import create_app


def test_health_and_frontend_are_available(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["semantic_vlm"]["configured"] is False
        page = client.get("/")
        assert page.status_code == 200
        assert "Football Intelligence" in page.text


def test_bundled_roster_can_be_loaded_and_custom_roster_is_validated(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        rows = client.get("/api/v1/rosters").json()
        assert len(rows) == 1
        assert {team["short"] for team in rows[0]["teams"]} == {"ARG", "FRA"}
        roster = client.get(f"/api/v1/rosters/{rows[0]['roster_id']}").json()
        assert roster["teams"][0]["players"]["10"] == "Messi"
        assert len(roster["teams"][0]["starting_lineup"]) == 11

        roster["name"] = "Edited final roster"
        saved = client.post("/api/v1/rosters", json=roster)
        assert saved.status_code == 201
        assert saved.json()["roster_id"] != rows[0]["roster_id"]

        roster["teams"][0]["goalkeepers"] = ["99"]
        invalid = client.post("/api/v1/rosters", json=roster)
        assert invalid.status_code == 422


def _match_on_disk(root, name="qatar-final"):
    import json

    directory = root / name
    directory.mkdir(parents=True)
    (directory / "match_report.json").write_text(
        json.dumps(
            {
                "title": "Аргентина — Франция",
                "clip_frames": 750,
                "frames_with_game_state": 250,
                "players": [{"track": 1}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (directory / "clip_web.mp4").write_bytes(b"\x00\x01video bytes")
    return directory


def test_the_viewer_lists_serves_and_streams_a_mounted_match(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    _match_on_disk(reports)
    monkeypatch.setenv("FI_REPORTS_DIR", str(reports))

    with TestClient(create_app(tmp_path / "runtime")) as client:
        listed = client.get("/api/v1/reports").json()
        assert [item["id"] for item in listed] == ["qatar-final"]
        assert listed[0]["title"] == "Аргентина — Франция"
        assert listed[0]["has_video"] is True

        page = client.get("/matches/qatar-final")
        assert page.status_code == 200
        assert "__MATCH_DATA__" not in page.text
        assert "/api/v1/reports/qatar-final/video" in page.text
        assert "<title>Аргентина — Франция</title>" in page.text

        # the clip streams, and it answers range requests so seeking works
        ranged = client.get(
            "/api/v1/reports/qatar-final/video", headers={"Range": "bytes=2-6"}
        )
        assert ranged.status_code == 206
        assert ranged.content == b"video"


def test_an_unknown_or_unsafe_match_id_is_refused(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    _match_on_disk(reports)
    monkeypatch.setenv("FI_REPORTS_DIR", str(reports))

    with TestClient(create_app(tmp_path / "runtime")) as client:
        assert client.get("/matches/does-not-exist").status_code == 404
        assert client.get("/api/v1/reports/..%2F..%2Fetc/video").status_code == 404


def test_the_older_lab_frontend_is_still_reachable(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        lab = client.get("/lab")
        assert lab.status_code == 200
        assert "EVIDENCE LAB" in lab.text


def test_the_backend_check_reports_what_is_missing(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setenv("FI_REPORTS_DIR", str(reports))
    monkeypatch.setenv("FI_CORE_URL", "http://127.0.0.1:9")  # nothing listens there

    with TestClient(create_app(tmp_path / "runtime")) as client:
        info = client.get("/api/v1/analysis/backend").json()
        assert info["reachable"] is False
        assert info["reports_writable"] is True


def test_a_run_cannot_start_without_a_video(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setenv("FI_REPORTS_DIR", str(reports))

    with TestClient(create_app(tmp_path / "runtime")) as client:
        response = client.post("/api/v1/analyses", json={"video": str(tmp_path / "nope.mp4")})
        assert response.status_code == 404


def test_a_broken_roster_is_refused_before_any_gpu_time(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setenv("FI_REPORTS_DIR", str(reports))
    video = tmp_path / "match.mp4"
    video.write_bytes(b"not really a video")

    roster = {
        "name": "Тест",
        "teams": [
            {
                "team": "A",
                "short": "AAA",
                "players": {"10": "Ten"},
                "goalkeepers": ["99"],  # not in the squad
                "starting_lineup": ["10"],
            }
        ],
    }
    with TestClient(create_app(tmp_path / "runtime")) as client:
        response = client.post(
            "/api/v1/analyses", json={"video": str(video), "roster": roster}
        )
        assert response.status_code == 422


def test_analyses_are_listed_and_can_be_fetched_by_id(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setenv("FI_REPORTS_DIR", str(reports))
    monkeypatch.setenv("FI_CORE_URL", "http://127.0.0.1:9")
    video = tmp_path / "match.mp4"
    video.write_bytes(b"not really a video")

    with TestClient(create_app(tmp_path / "runtime")) as client:
        started = client.post(
            "/api/v1/analyses", json={"video": str(video), "title": "Проверка"}
        )
        assert started.status_code == 202
        run_id = started.json()["id"]
        assert client.get(f"/api/v1/analyses/{run_id}").status_code == 200
        assert any(item["id"] == run_id for item in client.get("/api/v1/analyses").json())
        assert client.get("/api/v1/analyses/nope").status_code == 404


def test_an_interrupted_run_is_offered_and_can_be_resumed(tmp_path, monkeypatch):
    import json

    reports = tmp_path / "reports"
    (reports / "halfway" / "chunks").mkdir(parents=True)
    video = tmp_path / "match.mp4"
    video.write_bytes(b"not really a video")
    (reports / "halfway" / "plan.json").write_text(
        json.dumps(
            {
                "screen": {"playable_s": 120.0, "duration_s": 200.0},
                "chunks": [{"name": "chunk0000"}, {"name": "chunk0001"}],
                "video": str(video),
                "title": "После перезагрузки",
            }
        ),
        encoding="utf-8",
    )
    (reports / "halfway" / "chunks" / "chunk0000.json").write_text('{"predictions": []}')
    monkeypatch.setenv("FI_REPORTS_DIR", str(reports))
    monkeypatch.setenv("FI_CORE_URL", "http://127.0.0.1:9")

    with TestClient(create_app(tmp_path / "runtime")) as client:
        rows = client.get("/api/v1/analyses/interrupted").json()
        assert [row["id"] for row in rows] == ["halfway"]
        assert rows[0]["chunks_done"] == 1 and rows[0]["chunks_total"] == 2

        resumed = client.post("/api/v1/analyses/halfway/resume")
        assert resumed.status_code == 202
        assert resumed.json()["id"] == "halfway"
        assert client.post("/api/v1/analyses/nope/resume").status_code == 404


def test_detailed_statistics_endpoint_keeps_legacy_report_intact(tmp_path, monkeypatch):
    import json

    directory = tmp_path / "reports" / "test-only-match"
    directory.mkdir(parents=True)
    report = {"fps": 25, "first_frame": 1, "last_frame": 100, "players": [],
              "teams": {"left": {}, "right": {}}, "events": [], "title": "Test"}
    saved = directory / "match_report.json"
    saved.write_text(json.dumps(report), encoding="utf-8")
    original = saved.read_bytes()
    monkeypatch.setenv("FI_REPORTS_DIR", str(directory.parent))
    with TestClient(create_app(tmp_path / "runtime")) as client:
        response = client.get("/api/v1/reports/test-only-match/statistics")
        assert response.status_code == 200
        assert response.json()["schema_version"] == "1.0.0"
        assert response.json()["teams"]["left"]["shots"]["total"]["value"] is None
        assert client.get("/api/v1/reports/missing/statistics").status_code == 404
        assert client.get("/api/v1/reports/test-only-match").json() == report
        page = client.get("/matches/test-only-match")
        assert 'id="export-json"' in page.text
    assert saved.read_bytes() == original
