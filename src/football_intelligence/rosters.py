"""Validated match rosters and local roster storage."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator


class TeamRoster(BaseModel):
    team: str = Field(min_length=1, max_length=80)
    short: str = Field(pattern=r"^[A-Za-z0-9]{2,5}$")
    players: dict[str, str] = Field(min_length=1, max_length=40)
    goalkeepers: list[str] = Field(default_factory=list, max_length=8)
    starting_lineup: list[str] = Field(default_factory=list, max_length=11)

    @field_validator("players")
    @classmethod
    def validate_players(cls, players: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for raw_number, raw_name in players.items():
            number = str(raw_number).strip()
            if not number.isdigit() or not 0 <= int(number) <= 99:
                raise ValueError(f"invalid jersey number: {raw_number!r}")
            name = raw_name.strip()
            if not name:
                raise ValueError(f"empty player name for jersey {number}")
            normalized[str(int(number))] = name
        return normalized

    @model_validator(mode="after")
    def validate_goalkeepers(self) -> Self:
        self.short = self.short.upper()
        self.goalkeepers = [str(int(number)) for number in self.goalkeepers]
        self.starting_lineup = [str(int(number)) for number in self.starting_lineup]
        missing = sorted(set(self.goalkeepers) - set(self.players))
        if missing:
            raise ValueError(f"goalkeeper numbers absent from players: {missing}")
        missing_starters = sorted(set(self.starting_lineup) - set(self.players))
        if missing_starters:
            raise ValueError(f"starting lineup numbers absent from players: {missing_starters}")
        if self.starting_lineup and len(set(self.starting_lineup)) != 11:
            raise ValueError("starting_lineup must contain 11 distinct jersey numbers")
        return self


class MatchRoster(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    note: str = Field(default="", max_length=1000)
    teams: list[TeamRoster] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def teams_are_distinct(self) -> Self:
        shorts = [team.short for team in self.teams]
        names = [team.team.casefold() for team in self.teams]
        if len(set(shorts)) != 2 or len(set(names)) != 2:
            raise ValueError("the two roster teams must be distinct")
        return self

    def perception_payload(self, use_starting_lineup: bool = False) -> dict:
        payload = self.model_dump()
        for team in payload["teams"]:
            lineup = team.pop("starting_lineup", [])
            if use_starting_lineup and lineup:
                team["players"] = {
                    number: name
                    for number, name in team["players"].items()
                    if number in lineup
                }
                team["goalkeepers"] = [
                    number for number in team["goalkeepers"] if number in lineup
                ]
        return payload


class RosterStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def roster_id(roster: MatchRoster) -> str:
        canonical = roster.model_dump_json(exclude_none=True)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
        return f"roster-{digest}"

    def save(self, roster: MatchRoster) -> dict:
        roster_id = self.roster_id(roster)
        target = self.root / f"{roster_id}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(roster.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(target)
        return self.describe(roster_id, roster)

    def load(self, roster_id: str) -> MatchRoster:
        if not roster_id.startswith("roster-") or not roster_id[7:].isalnum():
            raise KeyError(roster_id)
        target = self.root / f"{roster_id}.json"
        if not target.is_file():
            raise KeyError(roster_id)
        return MatchRoster.model_validate_json(target.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        rosters: dict[str, tuple[str, MatchRoster]] = {}
        for path in sorted(self.root.glob("roster-*.json")):
            try:
                roster = MatchRoster.model_validate_json(path.read_text(encoding="utf-8"))
                identity = _roster_identity_without_lineup(roster)
                previous = rosters.get(identity)
                if previous is None or _lineup_size(roster) > _lineup_size(previous[1]):
                    rosters[identity] = (path.stem, roster)
            except (OSError, ValueError):
                continue
        return [
            self.describe(roster_id, roster)
            for roster_id, roster in sorted(rosters.values(), key=lambda row: row[1].name)
        ]

    @staticmethod
    def describe(roster_id: str, roster: MatchRoster) -> dict:
        return {
            "roster_id": roster_id,
            "name": roster.name,
            "teams": [
                {"team": team.team, "short": team.short, "players": len(team.players)}
                for team in roster.teams
            ],
        }


def install_bundled_rosters(store: RosterStore) -> None:
    package = files("football_intelligence").joinpath("roster_data")
    for resource in package.iterdir():
        if resource.name.endswith(".json"):
            roster = MatchRoster.model_validate(json.loads(resource.read_text(encoding="utf-8")))
            store.save(roster)


def _roster_identity_without_lineup(roster: MatchRoster) -> str:
    payload = roster.model_dump()
    for team in payload["teams"]:
        team.pop("starting_lineup", None)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _lineup_size(roster: MatchRoster) -> int:
    return sum(len(team.starting_lineup) for team in roster.teams)
