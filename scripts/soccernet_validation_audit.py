"""Audit a SoccerNet GSR validation archive without touching frozen test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--extract-sequence")
    parser.add_argument("--extract-root", type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    roles: Counter[str] = Counter()
    jersey_annotations = 0
    team_annotations = 0
    player_annotations = 0
    bbox_heights: list[float] = []
    tracks: dict[tuple[str, str], dict[str, bool]] = defaultdict(
        lambda: {"jersey": False, "team": False}
    )
    sequence_rows: list[dict] = []
    with zipfile.ZipFile(archive) as bundle:
        labels = sorted(
            name for name in bundle.namelist() if name.endswith("/Labels-GameState.json")
        )
        for label_name in labels:
            document = json.loads(bundle.read(label_name))
            info = document["info"]
            sequence = info["name"]
            for annotation in document["annotations"]:
                attributes = annotation.get("attributes", {})
                role = attributes.get("role", "ball")
                roles[role] += 1
                if role not in {"player", "goalkeeper"}:
                    continue
                player_annotations += 1
                key = (sequence, str(annotation["track_id"]))
                jersey = attributes.get("jersey")
                team = attributes.get("team")
                if jersey is not None:
                    jersey_annotations += 1
                    tracks[key]["jersey"] = True
                if team is not None:
                    team_annotations += 1
                    tracks[key]["team"] = True
                bbox_heights.append(annotation["bbox_image"]["h"] / 1080)
            sequence_rows.append(
                {
                    "sequence": sequence,
                    "version": str(info["version"]),
                    "frames": int(info["seq_length"]),
                    "fps": int(info["frame_rate"]),
                    "annotations": len(document["annotations"]),
                    "action_class": info.get("action_class"),
                    "visibility": info.get("visibility"),
                }
            )
        if args.extract_sequence:
            if not args.extract_root:
                raise ValueError("--extract-root is required with --extract-sequence")
            prefix = f"{args.extract_sequence}/"
            members = [item for item in bundle.infolist() if item.filename.startswith(prefix)]
            if not members:
                raise ValueError(f"sequence absent from archive: {args.extract_sequence}")
            bundle.extractall(args.extract_root.resolve(), members=members)

    player_tracks = len(tracks)
    jersey_tracks = sum(track["jersey"] for track in tracks.values())
    team_tracks = sum(track["team"] for track in tracks.values())
    result = {
        "run_id": f"soccernet_valid_audit_{datetime.now(UTC):%Y%m%dT%H%M%SZ}",
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": "SoccerNet Game State Reconstruction",
        "split": "valid",
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": file_hash(archive),
        "sequences": len(sequence_rows),
        "frames": sum(row["frames"] for row in sequence_rows),
        "annotations": sum(row["annotations"] for row in sequence_rows),
        "versions": dict(Counter(row["version"] for row in sequence_rows)),
        "roles": dict(roles),
        "player_or_goalkeeper_annotations": player_annotations,
        "jersey_annotation_coverage": jersey_annotations / player_annotations,
        "team_annotation_coverage": team_annotations / player_annotations,
        "player_or_goalkeeper_tracks": player_tracks,
        "jersey_track_coverage": jersey_tracks / player_tracks,
        "team_track_coverage": team_tracks / player_tracks,
        "bbox_height_normalized": {
            "median": statistics.median(bbox_heights),
            "p10": statistics.quantiles(bbox_heights, n=10)[0],
            "p90": statistics.quantiles(bbox_heights, n=10)[8],
        },
        "sequence_details": sequence_rows,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "sequence_details"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
