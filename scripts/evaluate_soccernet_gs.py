"""Evaluate Football Core predictions with the pinned official GS-HOTA runner."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--tracker-name", default="football-core")
    parser.add_argument("--soccernet-root", type=Path, default=Path("data/soccernet"))
    parser.add_argument(
        "--evaluator-root",
        type=Path,
        default=Path("data/tools/sn-trackeval"),
    )
    parser.add_argument("--use-roles", choices=("True", "False"), default="True")
    parser.add_argument("--use-teams", choices=("True", "False"), default="True")
    parser.add_argument(
        "--use-jersey-numbers", choices=("True", "False"), default="True"
    )
    args = parser.parse_args()

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("predictions JSON must contain a non-empty 'predictions' list")

    destination = (
        args.soccernet_root
        / "eval"
        / "trackers"
        / "SoccerNetGS-valid"
        / args.tracker_name
        / "data"
        / f"{args.sequence}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.predictions, destination)

    runner = args.evaluator_root / "scripts" / "run_soccernet_gs.py"
    command = [
        sys.executable,
        str(runner),
        "--GT_FOLDER",
        str(args.soccernet_root),
        "--TRACKERS_FOLDER",
        str(args.soccernet_root / "eval" / "trackers"),
        "--TRACKERS_TO_EVAL",
        args.tracker_name,
        "--SPLIT_TO_EVAL",
        "valid",
        "--SEQ_INFO",
        args.sequence,
        "--USE_PARALLEL",
        "False",
        "--PRINT_ONLY_COMBINED",
        "True",
        "--PLOT_CURVES",
        "False",
        "--USE_ROLES",
        args.use_roles,
        "--USE_TEAMS",
        args.use_teams,
        "--USE_JERSEY_NUMBERS",
        args.use_jersey_numbers,
        "--IGNORE_BALL",
        "True",
    ]
    subprocess.run(command, check=True)

    summary = (
        args.soccernet_root
        / "eval"
        / "trackers"
        / "SoccerNetGS-valid"
        / args.tracker_name
        / "person_summary.txt"
    )
    if not summary.is_file():
        raise FileNotFoundError(f"official evaluator did not create {summary}")
    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
