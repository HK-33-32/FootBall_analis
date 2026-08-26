from football_intelligence.domain import TrackletMemory
from football_intelligence.identity import GlobalIdentitySolver


def track(tracklet_id: str, first: int, last: int, jersey: int, team: str = "home"):
    return TrackletMemory(
        tracklet_id=tracklet_id,
        match_id="m1",
        team_posterior={team: 1.0},
        jersey_posterior={jersey: 1.0},
        first_seen_ms=first,
        last_seen_ms=last,
    )


def test_solver_links_non_overlapping_same_identity_without_reid():
    players, decisions = GlobalIdentitySolver().solve(
        [track("a", 0, 1000, 7), track("b", 2000, 3000, 7)]
    )
    assert len(players) == 1
    assert players[0].tracklet_ids == ["a", "b"]
    assert any(decision.merged for decision in decisions)


def test_solver_never_links_simultaneous_players_with_same_jersey():
    players, decisions = GlobalIdentitySolver().solve(
        [track("a", 0, 2000, 7), track("b", 1000, 3000, 7)]
    )
    assert len(players) == 2
    assert decisions[0].reasons == ("temporal_overlap",)


def test_solver_preserves_high_confidence_jersey_conflict():
    players, _ = GlobalIdentitySolver().solve([track("a", 0, 1000, 7), track("b", 2000, 3000, 10)])
    assert len(players) == 2


def test_solver_never_uses_team_as_the_only_identity_evidence():
    left = track("a", 0, 1000, 7)
    right = track("b", 2000, 3000, 10).model_copy(update={"jersey_posterior": {}})
    players, decisions = GlobalIdentitySolver().solve([left, right])
    assert len(players) == 2
    assert "team_only_is_not_identity_evidence" in decisions[0].reasons


def test_solver_quarantines_globally_collapsed_team_evidence():
    memories = [track("target-a", 0, 1000, 7, "left"), track("target-b", 2000, 3000, 7, "right")]
    memories.extend(
        track(f"filler-{number}", 0, 3000, number, "left") for number in range(20, 27)
    )
    solver = GlobalIdentitySolver()

    audit = solver.audit_team_evidence(memories)
    players, decisions = solver.solve(memories)

    assert audit.suppressed is True
    assert audit.shares == {"left": 8 / 9, "right": 1 / 9}
    assert any(
        decision.merged
        and {decision.left_tracklet_id, decision.right_tracklet_id} == {"target-a", "target-b"}
        for decision in decisions
    )
    merged = next(player for player in players if len(player.tracklet_ids) == 2)
    assert merged.team is None
