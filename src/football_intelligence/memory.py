"""SQLite Match Memory: structured facts first, source evidence always resolvable."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .domain import GlobalPlayerMemory, ModelRun, SemanticEvent

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS players (
  global_player_id TEXT PRIMARY KEY,
  match_id TEXT NOT NULL,
  team TEXT,
  jersey_number INTEGER,
  confidence REAL NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
  evidence_id TEXT PRIMARY KEY,
  match_id TEXT NOT NULL,
  source_uri TEXT NOT NULL,
  start_ms INTEGER NOT NULL,
  end_ms INTEGER NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  match_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  actor_player_id TEXT,
  target_player_id TEXT,
  start_ms INTEGER NOT NULL,
  end_ms INTEGER NOT NULL,
  confidence REAL NOT NULL,
  insufficient_evidence INTEGER NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS event_evidence (
  event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
  evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id),
  PRIMARY KEY(event_id, evidence_id)
);
CREATE INDEX IF NOT EXISTS idx_events_match_time ON events(match_id, start_ms);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(match_id, event_type);
CREATE INDEX IF NOT EXISTS idx_players_match ON players(match_id);
"""


class MatchMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def put_run(self, run: ModelRun) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO runs(run_id, payload_json) VALUES (?, ?)",
                (run.run_id, run.model_dump_json()),
            )

    def put_players(self, players: list[GlobalPlayerMemory]) -> None:
        with self.connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO players VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        player.global_player_id,
                        player.match_id,
                        player.team,
                        player.jersey_number,
                        player.identity_confidence,
                        player.model_dump_json(),
                    )
                    for player in players
                ],
            )

    def put_events(self, events: list[SemanticEvent]) -> None:
        with self.connect() as connection:
            for event in events:
                for evidence in event.evidence:
                    connection.execute(
                        "INSERT OR REPLACE INTO evidence VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            evidence.evidence_id,
                            evidence.match_id,
                            evidence.source_uri,
                            evidence.start_ms,
                            evidence.end_ms,
                            evidence.model_dump_json(),
                        ),
                    )
                connection.execute(
                    "INSERT OR REPLACE INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.event_id,
                        event.match_id,
                        event.primary_event.value,
                        event.actor_global_player_id,
                        event.target_global_player_id,
                        event.start_ms,
                        event.end_ms,
                        event.confidence,
                        int(event.insufficient_evidence),
                        event.model_dump_json(),
                    ),
                )
                connection.executemany(
                    "INSERT OR REPLACE INTO event_evidence VALUES (?, ?)",
                    [(event.event_id, evidence.evidence_id) for evidence in event.evidence],
                )

    def events(self, match_id: str, event_type: str | None = None) -> list[dict]:
        query = "SELECT payload_json FROM events WHERE match_id = ?"
        params: list[object] = [match_id]
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        query += " ORDER BY start_ms"
        with self.connect() as connection:
            return [json.loads(row[0]) for row in connection.execute(query, params)]

    def players(self, match_id: str) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM players WHERE match_id = ? ORDER BY team, jersey_number",
                (match_id,),
            )
            return [json.loads(row[0]) for row in rows]

    def evidence(self, evidence_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM evidence WHERE evidence_id = ?", (evidence_id,)
            ).fetchone()
            return json.loads(row[0]) if row else None

    def statistics(self, match_id: str) -> dict:
        """Only accepted semantic events contribute; unknown/abstained remain coverage."""
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT event_type, COUNT(*) AS n
                   FROM events WHERE match_id = ? AND insufficient_evidence = 0
                   GROUP BY event_type ORDER BY event_type""",
                (match_id,),
            ).fetchall()
            total = connection.execute(
                "SELECT COUNT(*) FROM events WHERE match_id = ?", (match_id,)
            ).fetchone()[0]
            accepted = sum(row[1] for row in rows)
        return {
            "match_id": match_id,
            "source": "semantic_events",
            "event_schema_version": "1.0.0",
            "counts": {row[0]: row[1] for row in rows},
            "coverage": accepted / total if total else 0.0,
            "accepted": accepted,
            "total": total,
        }
