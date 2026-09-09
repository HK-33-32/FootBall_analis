"""The Event Ledger: the single source of truth every statistic is read from.

Nothing downstream may count something that is not an event here, and every
event carries where it came from and how much it can be trusted.  Two numbers
are kept apart on purpose: `confidence` is how sure we are the event happened,
`player_confidence` is how sure we are it was *this* player — a pass can be
certain while the shirt number on the passer is a coin flip.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable


@dataclass
class Clock:
    """Every event is stamped in each time base we can honestly provide."""
    frame: int                    # frame index inside the analysed segment
    video_time_s: float           # seconds from the start of the source video
    segment_time_s: float         # seconds from the start of the analysed clip
    match_clock: str | None = None   # "MM:SS" once a kickoff offset is known
    period: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Event:
    event_id: str
    type: str
    clock: Clock
    duration_s: float = 0.0
    player_id: str | None = None
    player_confidence: float = 0.0
    team: str | None = None
    related_player_id: str | None = None
    outcome: str | None = None
    start_xy: tuple[float, float] | None = None
    end_xy: tuple[float, float] | None = None
    confidence: float = 1.0
    source: str = "geometry"      # geometry | spotter | fusion | manual
    track_id: int | None = None
    related_track_id: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["clock"] = self.clock.as_dict()
        return d


class EventLedger:
    """An append-only list of events plus the metadata needed to read them."""

    def __init__(self, meta: dict | None = None):
        self.meta: dict = meta or {}
        self.events: list[Event] = []
        self.players: dict[str, dict] = {}
        self._seq = 0

    def next_id(self, prefix: str) -> str:
        self._seq += 1
        return "%s_%06d" % (prefix, self._seq)

    def add(self, event: Event) -> Event:
        self.events.append(event)
        return event

    def extend(self, events: Iterable[Event]) -> None:
        self.events.extend(events)

    def sort(self) -> None:
        self.events.sort(key=lambda e: (e.clock.frame, e.type))

    def by_player(self, player_id: str) -> list[Event]:
        return [e for e in self.events if e.player_id == player_id]

    # --- persistence --------------------------------------------------------
    def to_json(self, path: str) -> None:
        payload = {
            "meta": self.meta,
            "players": self.players,
            "events": [e.as_dict() for e in self.events],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, path: str) -> "EventLedger":
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        ledger = cls(payload.get("meta"))
        ledger.players = payload.get("players", {})
        for raw in payload.get("events", []):
            clock = Clock(**raw.pop("clock"))
            start = raw.pop("start_xy", None)
            end = raw.pop("end_xy", None)
            ledger.events.append(Event(
                clock=clock,
                start_xy=tuple(start) if start else None,
                end_xy=tuple(end) if end else None,
                **raw))
        return ledger

    def to_sqlite(self, path: str) -> None:
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        cur.executescript("""
        DROP TABLE IF EXISTS events;
        DROP TABLE IF EXISTS players;
        DROP TABLE IF EXISTS meta;
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY, type TEXT, frame INTEGER,
            video_time_s REAL, segment_time_s REAL, match_clock TEXT,
            period INTEGER, duration_s REAL,
            player_id TEXT, player_confidence REAL, team TEXT,
            related_player_id TEXT, outcome TEXT,
            start_x REAL, start_y REAL, end_x REAL, end_y REAL,
            confidence REAL, source TEXT, track_id INTEGER,
            related_track_id INTEGER, attributes TEXT);
        CREATE INDEX idx_events_player ON events(player_id);
        CREATE INDEX idx_events_type ON events(type);
        CREATE INDEX idx_events_frame ON events(frame);
        CREATE TABLE players (
            player_id TEXT PRIMARY KEY, team TEXT, jersey TEXT,
            name TEXT, role TEXT, attribution_confidence REAL, track_ids TEXT);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        """)
        cur.executemany(
            "INSERT INTO events VALUES (%s)" % ",".join("?" * 22),
            [(e.event_id, e.type, e.clock.frame, e.clock.video_time_s,
              e.clock.segment_time_s, e.clock.match_clock, e.clock.period,
              e.duration_s, e.player_id, e.player_confidence, e.team,
              e.related_player_id, e.outcome,
              e.start_xy[0] if e.start_xy else None,
              e.start_xy[1] if e.start_xy else None,
              e.end_xy[0] if e.end_xy else None,
              e.end_xy[1] if e.end_xy else None,
              e.confidence, e.source, e.track_id, e.related_track_id,
              json.dumps(e.attributes, ensure_ascii=False))
             for e in self.events])
        cur.executemany(
            "INSERT INTO players VALUES (?,?,?,?,?,?,?)",
            [(pid, p.get("team"), p.get("jersey"), p.get("name"), p.get("role"),
              p.get("attribution_confidence", 0.0),
              json.dumps(p.get("track_ids", [])))
             for pid, p in self.players.items()])
        cur.executemany("INSERT INTO meta VALUES (?,?)",
                        [(k, json.dumps(v, ensure_ascii=False))
                         for k, v in self.meta.items()])
        conn.commit()
        conn.close()
