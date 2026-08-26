import json
from importlib.resources import files

from football_intelligence.rosters import MatchRoster, RosterStore


def bundled_roster() -> MatchRoster:
    source = files("football_intelligence").joinpath(
        "roster_data/wc2022_final_arg_fra.json"
    )
    return MatchRoster.model_validate(json.loads(source.read_text(encoding="utf-8")))


def test_starting_lineup_payload_narrows_candidates_without_mutating_roster():
    roster = bundled_roster()
    payload = roster.perception_payload(use_starting_lineup=True)
    assert len(payload["teams"][0]["players"]) == 11
    assert set(payload["teams"][0]["players"]) == set(roster.teams[0].starting_lineup)
    assert len(roster.teams[0].players) == 26


def test_full_squad_payload_preserves_all_players():
    roster = bundled_roster()
    payload = roster.perception_payload(use_starting_lineup=False)
    assert len(payload["teams"][0]["players"]) == 26


def test_roster_list_collapses_legacy_version_without_lineups(tmp_path):
    store = RosterStore(tmp_path)
    roster = bundled_roster()
    complete = store.save(roster)
    legacy = roster.model_copy(
        update={
            "teams": [team.model_copy(update={"starting_lineup": []}) for team in roster.teams]
        }
    )
    store.save(legacy)

    rows = store.list()

    assert len(rows) == 1
    assert rows[0]["roster_id"] == complete["roster_id"]
