"""Check the numbers a run produced against the squads that actually played.

Without ground truth there is no accuracy to measure, but a squad list still
gives one thing for free: a number nobody in either squad wears is provably
wrong. This reports that, plus which squad numbers were never found, and names
the players behind the numbers that were.

It is a report, not a constraint -- nothing here changes the predictions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_roster(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def squad_numbers(roster: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {team["team"]: dict(team.get("players") or {}) for team in roster["teams"]}


def starting_numbers(roster: dict[str, Any]) -> dict[str, set[str]]:
    return {
        team["team"]: {str(number) for number in team.get("starting_lineup") or []}
        for team in roster["teams"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    roster = load_roster(args.roster)
    squads = squad_numbers(roster)
    starters = starting_numbers(roster)
    everyone = {number for players in squads.values() for number in players}

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["predictions"]
    per_team: dict[str, Counter] = defaultdict(Counter)
    identities: dict[str, set[int]] = defaultdict(set)
    for detection in predictions:
        attributes = detection.get("attributes") or {}
        if attributes.get("role") != "player":
            continue
        jersey = attributes.get("jersey")
        if jersey in (None, ""):
            continue
        side = attributes.get("team") or "unknown"
        per_team[side][str(jersey)] += 1
        identities[side].add(detection["track_id"])

    report: dict[str, Any] = {
        "roster": roster.get("name"),
        "predictions": str(args.predictions),
        "squads": {name: sorted(players, key=int) for name, players in squads.items()},
        "sides": {},
    }
    for side, numbers in sorted(per_team.items()):
        impossible = {n: c for n, c in numbers.items() if n not in everyone}
        report["sides"][side] = {
            "identities_with_a_number": len(identities[side]),
            "numbers_read": dict(sorted(numbers.items(), key=lambda kv: -kv[1])),
            "outside_both_squads": impossible,
            "detections_outside_both_squads": sum(impossible.values()),
            "detections_with_a_number": sum(numbers.values()),
        }

    read = {number for numbers in per_team.values() for number in numbers}
    for name, squad in squads.items():
        found = sorted(read & set(squad), key=int)
        report.setdefault("per_squad", {})[name] = {
            "numbers_found": {number: squad[number] for number in found},
            "starters_found": sorted(read & starters.get(name, set()), key=int),
            "starters_missing": sorted(starters.get(name, set()) - read, key=int),
        }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
