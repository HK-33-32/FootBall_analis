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
