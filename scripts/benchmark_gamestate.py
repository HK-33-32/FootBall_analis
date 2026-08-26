"""Benchmark the game-state refinement against raw perception on SoccerNet GSR valid clips.

For every sequence the script evaluates the perception output as-is, applies
:mod:`football_intelligence.gamestate` and evaluates again with the pinned
official GS-HOTA runner. Nothing here reads ground truth outside the evaluator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.gamestate import (  # noqa: E402
    JerseyTally,
    RefinementConfig,
    load_reid_embeddings,
    refine_predictions,
    sequence_frame_loader,
)


def find_reid_pickle(run_dir: Path, jobs_root: Path) -> Path | None:
    """Locate the perception tracklet dump that belongs to this run, if kept."""
    job_file = run_dir / "core_job.json"
    if not job_file.is_file():
        return None
    job_id = json.loads(job_file.read_text(encoding="utf-8")).get("id")
    if not job_id:
        return None
    candidates = sorted((jobs_root / str(job_id)).rglob("*.pkl"))
    return candidates[0] if candidates else None


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


METRIC_KEYS = (
    "HOTA",
    "DetA",
    "AssA",
    "DetRe",
    "DetPr",
    "AssRe",
    "AssPr",
    "LocA",
    "OWTA",
    "HOTA(0)",
    "LocA(0)",
    "HOTALocA(0)",
    "IDF1",
    "IDR",
    "IDP",
    "IDTP",
    "IDFN",
    "IDFP",
    "Dets",
    "GT_Dets",
    "IDs",
    "GT_IDs",
)


def evaluate(
    predictions: Path, sequence: str, tracker: str, soccernet_root: Path, evaluator_root: Path
) -> dict[str, float]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_soccernet_gs.py")),
        "--predictions",
        str(predictions),
        "--sequence",
        sequence,
        "--tracker-name",
        tracker,
        "--soccernet-root",
        str(soccernet_root),
        "--evaluator-root",
        str(evaluator_root),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if line.startswith("HOTA DetA AssA") and index + 1 < len(lines):
            values = lines[index + 1].split()
            return {key: float(value) for key, value in zip(METRIC_KEYS, values, strict=False)}
    raise RuntimeError(f"could not parse evaluator output for {sequence}/{tracker}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-template", default="soccernet_valid_{sequence}_core")
    parser.add_argument("--frames-root", type=Path, default=Path("data/soccernet/valid"))
    parser.add_argument("--soccernet-root", type=Path, default=Path("data/soccernet"))
    parser.add_argument("--evaluator-root", type=Path, default=Path("data/tools/sn-trackeval"))
    parser.add_argument("--output", type=Path, default=Path("runs/gamestate_benchmark.json"))
    parser.add_argument("--tag", default="refined")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--lightness-weight", type=float, default=0.0)
    parser.add_argument("--link-max-gap-frames", type=int, default=400)
    parser.add_argument("--link-max-distance-m", type=float, default=5.0)
    parser.add_argument("--no-link", action="store_true")
    parser.add_argument("--jersey-fill", action="store_true")
    parser.add_argument("--no-jersey-propagate", action="store_true")
    parser.add_argument("--no-reid", action="store_true")
    parser.add_argument("--no-identity-reads", action="store_true")
    parser.add_argument("--identity-read-weight", type=float, default=5.0)
    parser.add_argument("--identity-read-min-consensus", type=float, default=0.34)
    parser.add_argument("--no-number-exclusivity", action="store_true")
    parser.add_argument("--no-reconcile-roles", action="store_true")
    parser.add_argument("--reid-min-cosine", type=float, default=0.85)
    parser.add_argument("--reid-max-distance-m", type=float, default=15.0)
    parser.add_argument("--jobs-root", type=Path, default=Path("data/perception_benchmark/jobs"))
    args = parser.parse_args()

    config = RefinementConfig(
        frame_stride=args.frame_stride,
        lightness_weight=args.lightness_weight,
        link_tracklets=not args.no_link,
        link_max_gap_frames=args.link_max_gap_frames,
        link_max_distance_m=args.link_max_distance_m,
        jersey_fill=args.jersey_fill,
        jersey_propagate=not args.no_jersey_propagate,
        identity_read_weight=args.identity_read_weight,
        identity_read_min_consensus=args.identity_read_min_consensus,
        enforce_number_exclusivity=not args.no_number_exclusivity,
        reconcile_roles=not args.no_reconcile_roles,
        reid_min_cosine=args.reid_min_cosine,
        reid_max_distance_m=args.reid_max_distance_m,
    )
    results: dict[str, dict] = {}
    for sequence in args.sequences:
        run_dir = args.runs_root / args.run_template.format(sequence=sequence)
        if not (run_dir / "predictions.json").is_file():
            # the first reproduced run predates the {sequence} naming convention
            legacy = args.runs_root / args.run_template.format(
                sequence=sequence.replace("-", "").lower()
            )
            if (legacy / "predictions.json").is_file():
                run_dir = legacy
        source = run_dir / "predictions.json"
        if not source.is_file():
            print(f"skip {sequence}: {source} missing", file=sys.stderr)
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        frames = sequence_frame_loader(str(args.frames_root), sequence)
        reid = None if args.no_reid else find_reid_pickle(run_dir, args.jobs_root)
        started = time.perf_counter()
        embeddings = load_reid_embeddings(str(reid), payload["predictions"]) if reid else None
        reads_file = run_dir / "jersey_reads.json"
        reads = (
            load_identity_reads(reads_file)
            if reads_file.is_file() and not args.no_identity_reads
            else None
        )
        refined, report = refine_predictions(
            payload["predictions"], frames, config, embeddings, reads
        )
        report["reid_pickle"] = str(reid) if reid else None
        report["identity_reads"] = str(reads_file) if reads else None
        report["wall_time_s"] = round(time.perf_counter() - started, 2)

        target = run_dir / f"predictions_{args.tag}.json"
        target.write_text(json.dumps({**payload, "predictions": refined}), encoding="utf-8")
        (run_dir / f"refinement_{args.tag}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        results[sequence] = {
            "baseline": evaluate(
                source, sequence, f"core-{sequence}", args.soccernet_root, args.evaluator_root
            ),
            "refined": evaluate(
                target,
                sequence,
                f"{args.tag}-{sequence}",
                args.soccernet_root,
                args.evaluator_root,
            ),
            "refinement": report,
        }
        base = results[sequence]["baseline"]["HOTA"]
        best = results[sequence]["refined"]["HOTA"]
        print(f"{sequence}: GS-HOTA {base:.3f} -> {best:.3f}  ({best - base:+.3f})", flush=True)

    if results:
        summary = {}
        for stage in ("baseline", "refined"):
            summary[stage] = {
                key: round(sum(entry[stage][key] for entry in results.values()) / len(results), 3)
                for key in ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "IDF1")
            }
        payload = {
            "sequences": sorted(results),
            "config": asdict(config),
            "per_sequence": results,
            "macro_average": summary,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"written: {args.output}")


if __name__ == "__main__":
    main()
