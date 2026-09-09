"""Splitting a long interval into segments and stitching the results back.

The pipeline is built around SoccerNet clips of 30 seconds: every stage walks the
whole frame folder, and `refine_tracklets` compares every tracklet with every
other one.  A full match would need ~100 GB of extracted JPEGs and a quadratic
tracklet pass, so long intervals are processed one segment at a time: frames for
a segment are extracted, analysed, rendered and then deleted before the next
segment starts.  Disk and memory stay flat no matter how long the match is.

Track ids and frame numbers are shifted per segment so the merged JSON reads as
one continuous timeline.  Identities are not carried across a segment boundary —
the same player gets a new id there, the same way it happens at a camera cut.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def plan_segments(start: float, end: float, segment_seconds: float,
                  max_segments: int = 400) -> list[tuple[float, float]]:
    """Cut [start, end) into pieces of at most segment_seconds."""
    total = max(0.0, end - start)
    if segment_seconds <= 0 or total <= segment_seconds:
        return [(start, end)]

    count = int(total // segment_seconds) + (1 if total % segment_seconds > 0.5 else 0)
    count = max(1, min(count, max_segments))
    step = total / count
    bounds = []
    for index in range(count):
        seg_start = start + index * step
        seg_end = end if index == count - 1 else start + (index + 1) * step
        bounds.append((seg_start, seg_end))
    return bounds


def concat_videos(parts: list[Path], output: Path) -> None:
    """Join per-segment mp4s without re-encoding."""
    if not parts:
        raise RuntimeError("нет сегментов для сборки")
    if len(parts) == 1:
        parts[0].replace(output)
        return

    listing = output.parent / "segments.txt"
    listing.write_text(
        "".join("file '%s'\n" % p.as_posix() for p in parts), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
         "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c", "copy", "-movflags", "+faststart", str(output)],
        check=True)
    listing.unlink(missing_ok=True)


def merge_predictions(segment_files: list[tuple[Path, int, int]],
                      output: Path) -> dict:
    """Merge per-segment JSONs into one timeline.

    segment_files: (json path, frame offset, track id offset).
    Returns counters for the job log.
    """
    merged = []
    tracks = set()
    for path, frame_offset, track_offset in segment_files:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for pred in data.get("predictions", []):
            image_id = str(pred.get("image_id", "0"))
            local_frame = int(image_id[-6:]) if image_id[-6:].isdigit() else 0
            frame = local_frame + frame_offset
            track_id = int(pred.get("track_id", 0)) + track_offset
            pred["image_id"] = "3%s%06d" % (str(pred.get("video_id", "0")), frame)
            pred["frame"] = frame
            pred["track_id"] = track_id
            tracks.add(track_id)
            merged.append(pred)

    for index, pred in enumerate(merged):
        pred["id"] = str(index)
    output.write_text(json.dumps({"predictions": merged}, indent=2), encoding="utf-8")
    return {"predictions": len(merged), "tracks": len(tracks)}
