"""Refine perception predictions with tracklet-level team and jersey resolution."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.gamestate import (  # noqa: E402
    JerseyTally,
    RefinementConfig,
    load_reid_embeddings,
    refine_predictions,
    sequence_frame_loader,
)


def load_identity_reads(path: Path) -> dict[int, JerseyTally]:
    """Tallies written by ``scripts/reread_jerseys.py``, keyed by identity id."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(entry["identity"]): JerseyTally(
            votes=Counter({str(number): int(count) for number, count in entry["tally"].items()}),
            questions=int(entry.get("questions") or len(entry.get("answers") or ())),
        )
        for entry in payload.get("identities", [])
        if entry.get("tally")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frames-root", type=Path, default=Path("data/soccernet/valid"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--lightness-weight", type=float, default=0.0)
    parser.add_argument("--offside-rank", type=int, default=2)
    parser.add_argument("--jersey-min-votes", type=int, default=2)
    parser.add_argument("--jersey-min-share", type=float, default=0.34)
    parser.add_argument("--jersey-fill", action="store_true")
    parser.add_argument("--no-jersey-propagate", action="store_true")
    parser.add_argument("--link-tracklets", action="store_true")
    parser.add_argument("--link-max-gap-frames", type=int, default=400)
    parser.add_argument("--link-max-distance-m", type=float, default=5.0)
    parser.add_argument(
        "--reid-pickle",
        type=Path,
        default=None,
        help="perception tracklet dump with re-identification features",
    )
    parser.add_argument("--reid-min-cosine", type=float, default=0.85)
    parser.add_argument("--reid-max-distance-m", type=float, default=15.0)
    parser.add_argument(
        "--jersey-reads", type=Path, default=None, help="tallies from scripts/reread_jerseys.py"
    )
    parser.add_argument("--identity-read-weight", type=float, default=5.0)
    parser.add_argument("--identity-read-min-consensus", type=float, default=0.34)
    parser.add_argument("--no-number-exclusivity", action="store_true")
    parser.add_argument("--no-reconcile-roles", action="store_true")
    parser.add_argument("--no-teams", action="store_true", help="keep the incoming team labels")
    parser.add_argument("--no-jerseys", action="store_true", help="keep the incoming jersey labels")
    args = parser.parse_args()

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions = payload["predictions"]
    config = RefinementConfig(
        frame_stride=args.frame_stride,
        lightness_weight=args.lightness_weight,
        offside_rank=args.offside_rank,
        jersey_min_votes=args.jersey_min_votes,
        jersey_min_share=args.jersey_min_share,
        jersey_fill=args.jersey_fill,
        jersey_propagate=not args.no_jersey_propagate,
        link_tracklets=args.link_tracklets,
        link_max_gap_frames=args.link_max_gap_frames,
        link_max_distance_m=args.link_max_distance_m,
        identity_read_weight=args.identity_read_weight,
        identity_read_min_consensus=args.identity_read_min_consensus,
        enforce_number_exclusivity=not args.no_number_exclusivity,
        reconcile_roles=not args.no_reconcile_roles,
        reid_min_cosine=args.reid_min_cosine,
        reid_max_distance_m=args.reid_max_distance_m,
    )
    frames = sequence_frame_loader(str(args.frames_root), args.sequence)
    started = time.perf_counter()
    embeddings = (
        load_reid_embeddings(str(args.reid_pickle), predictions) if args.reid_pickle else None
    )
    reads = load_identity_reads(args.jersey_reads) if args.jersey_reads else None
    refined, report = refine_predictions(predictions, frames, config, embeddings, reads)
    report["wall_time_s"] = round(time.perf_counter() - started, 2)
    report["sequence"] = args.sequence
    report["source_predictions"] = str(args.predictions)

    if args.no_teams or args.no_jerseys:
        for original, item in zip(predictions, refined, strict=True):
            source = original.get("attributes") or {}
            if args.no_teams:
                item["attributes"]["team"] = source.get("team")
            if args.no_jerseys:
                item["attributes"]["jersey"] = source.get("jersey")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload["predictions"] = refined
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
