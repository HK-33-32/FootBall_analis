"""Cutting a long match into jobs the perception backend can actually finish.

The backend takes a clip and returns a game state; it was never meant to be
handed ninety minutes at once. Three things have to happen around it.

**Only the football goes in.** :mod:`football_intelligence.playability` decides
which stretches show the pitch; everything else never reaches the GPU.

**The work is bounded and resumable.** Each stretch is cut into chunks of at
most a minute or two, so a failure costs one chunk rather than a day, and a
rerun skips whatever is already on disk.

**The pieces are put back together.** Frame numbers are rebased onto the match
clock and track ids are made unique per chunk. Identity is *not* carried across
a chunk boundary: the same player entering the next chunk becomes a new
identity. That is deliberate rather than unfinished -- chunks are cut at the
edges of playable stretches wherever possible, which is exactly where the
broadcast already cut away and where the tracker had lost him anyway.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

Detection = dict[str, Any]


@dataclass(frozen=True)
class ChunkConfig:
    """How a match is divided into perception jobs."""

    max_chunk_s: float = 60.0
    min_chunk_s: float = 2.0
    fps: float = 25.0
    track_id_stride: int = 100_000


def plan_chunks(
    segments: Sequence[dict[str, Any]], config: ChunkConfig | None = None
) -> list[dict[str, Any]]:
    """Split playable stretches into bounded jobs, keeping the match clock."""
    config = config or ChunkConfig()
    chunks: list[dict[str, Any]] = []
    for segment in segments:
        start = float(segment["start_s"])
        end = float(segment["end_s"])
        length = end - start
        if length < config.min_chunk_s:
            continue
        pieces = max(1, int(-(-length // config.max_chunk_s)))  # ceil
        step = length / pieces
        for index in range(pieces):
            piece_start = start + index * step
            piece_end = min(end, piece_start + step)
            if piece_end - piece_start < config.min_chunk_s and chunks:
                chunks[-1]["end_s"] = round(piece_end, 3)
                chunks[-1]["duration_s"] = round(chunks[-1]["end_s"] - chunks[-1]["start_s"], 3)
                continue
            chunks.append(
                {
                    "index": len(chunks),
                    "start_s": round(piece_start, 3),
                    "end_s": round(piece_end, 3),
                    "duration_s": round(piece_end - piece_start, 3),
                    "start_frame": int(round(piece_start * config.fps)) + 1,
                }
            )
    for position, chunk in enumerate(chunks):
        chunk["index"] = position
        chunk["name"] = f"chunk{position:04d}"
    return chunks


def rebase(
    predictions: Iterable[Detection], chunk: dict[str, Any], config: ChunkConfig | None = None
) -> list[Detection]:
    """Move one chunk's detections onto the match clock and a private id range."""
    config = config or ChunkConfig()
    offset = int(chunk["start_frame"]) - 1
    prefix = int(chunk["index"]) * config.track_id_stride
    rebased: list[Detection] = []
    for detection in predictions:
        item = dict(detection)
        item["frame"] = int(detection["frame"]) + offset
        if item.get("track_id") is not None:
            item["track_id"] = int(detection["track_id"]) + prefix
        item["chunk"] = chunk["name"]
        rebased.append(item)
    return rebased


def merge_chunks(
    parts: Sequence[tuple[dict[str, Any], Sequence[Detection]]],
    config: ChunkConfig | None = None,
) -> dict[str, Any]:
    """Concatenate rebased chunks into one match-long prediction set."""
    config = config or ChunkConfig()
    merged: list[Detection] = []
    covered: list[dict[str, Any]] = []
    for chunk, predictions in parts:
        rows = rebase(predictions, chunk, config)
        merged.extend(rows)
        frames = [row["frame"] for row in rows]
        covered.append(
            {
                "name": chunk["name"],
                "start_s": chunk["start_s"],
                "end_s": chunk["end_s"],
                "detections": len(rows),
                "first_frame": min(frames) if frames else None,
                "last_frame": max(frames) if frames else None,
            }
        )
    merged.sort(key=lambda row: (row["frame"], row.get("track_id") or 0))
    return {
        "predictions": merged,
        "chunks": covered,
        "detections": len(merged),
        "identities": len(
            {row.get("track_id") for row in merged if row.get("track_id") is not None}
        ),
    }


__all__ = ["ChunkConfig", "merge_chunks", "plan_chunks", "rebase"]
