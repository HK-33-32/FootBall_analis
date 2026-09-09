"""Build the single JSON a match viewer needs: statistics plus playback timeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.calibration import (  # noqa: E402
    CalibrationConfig,
    drop_uncalibrated,
)
from football_intelligence.match_stats import (  # noqa: E402
    StatsConfig,
    match_statistics,
    timeline,
)
from football_intelligence.reporting import (  # noqa: E402
    ReportAnnotations,
    ReportConfig,
    detailed_report,
    report_with_duration,
)
from football_intelligence.trajectories import (  # noqa: E402
    TrajectoryConfig,
    clean_ball_track,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument(
        "--duration-ms", type=int, help="full source interval, including empty frames"
    )
    parser.add_argument("--roster", type=Path, default=None)
    parser.add_argument("--match-id", default="")
    parser.add_argument(
        "--video-source", default="", help="source video path/URI for event evidence"
    )
    parser.add_argument(
        "--event-annotations",
        type=Path,
        help="validated event annotations; replaces geometric candidates in detailed JSON",
    )
    parser.add_argument("--report-config", type=Path, help="JSON overrides for ReportConfig")
    parser.add_argument(
        "--statistics-output", type=Path, help="also export standalone detailed statistics"
    )
    parser.add_argument("--no-timeline", action="store_true")
    parser.add_argument(
        "--score-timeline",
        type=Path,
        default=None,
        help="scoreboard readings, as written by scripts/read_scoreboard.py. When given "
        "it decides which goals are real: a ball passing behind the net looks the same "
        "from one camera, and the score does not.",
    )
    parser.add_argument(
        "--cards",
        type=Path,
        default=None,
        help="cards read by scripts/read_cards.py at the stoppages this report found",
    )
    parser.add_argument(
        "--keep-uncalibrated",
        action="store_true",
        help="publish positions from frames whose homography fails its sanity tests",
    )
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["predictions"]
    calibration = {"frames_rejected": 0}
    if not args.keep_uncalibrated:
        predictions, calibration = drop_uncalibrated(predictions, CalibrationConfig())
    stats_config = StatsConfig(fps=args.fps)
    trajectory_config = TrajectoryConfig(fps=args.fps)
    score_timeline = None
    if args.score_timeline and args.score_timeline.is_file():
        payload = json.loads(args.score_timeline.read_text(encoding="utf-8"))
        score_timeline = payload["readings"] if isinstance(payload, dict) else payload
    cards = None
    if args.cards and args.cards.is_file():
        payload = json.loads(args.cards.read_text(encoding="utf-8"))
        cards = payload["cards"] if isinstance(payload, dict) else payload
    report = match_statistics(
        predictions,
        stats_config,
        trajectory_config,
        score_timeline=score_timeline,
        cards=cards,
    )
    report = report_with_duration(report, args.duration_ms)
    report["title"] = args.title or args.predictions.parent.name
    report["calibration"] = {key: value for key, value in calibration.items() if key != "rejected"}
    report["source"] = str(args.predictions)
    with args.predictions.open("rb") as prediction_stream:
        report["predictions_sha256"] = hashlib.file_digest(prediction_stream, "sha256").hexdigest()
    if args.video_source:
        report["video_source"] = args.video_source

    if args.roster and args.roster.is_file():
        roster = json.loads(args.roster.read_text(encoding="utf-8"))
        report["roster"] = {
            "name": roster.get("name"),
            "teams": [
                {
                    "team": team["team"],
                    "short": team.get("short"),
                    "players": team.get("players") or {},
                    "starting_lineup": team.get("starting_lineup") or [],
                }
                for team in roster.get("teams", [])
            ],
        }

    annotations = (
        ReportAnnotations.model_validate_json(args.event_annotations.read_text("utf-8"))
        if args.event_annotations
        else None
    )
    report_config = (
        ReportConfig.model_validate_json(args.report_config.read_text("utf-8"))
        if args.report_config
        else ReportConfig()
    )
    report["detailed_statistics"] = detailed_report(
        report,
        annotations=annotations,
        match_id=args.match_id or args.predictions.parent.name,
        config=report_config,
        predictions=predictions,
    )
    if args.statistics_output:
        args.statistics_output.parent.mkdir(parents=True, exist_ok=True)
        args.statistics_output.write_text(
            json.dumps(
                report["detailed_statistics"], ensure_ascii=False, indent=2, allow_nan=False
            ),
            encoding="utf-8",
        )

    if not args.no_timeline:
        ball = [
            (
                int(detection["frame"]),
                (
                    float(detection["bbox_pitch"]["x_bottom_middle"]),
                    float(detection["bbox_pitch"]["y_bottom_middle"]),
                ),
            )
            for detection in predictions
            if (detection.get("attributes") or {}).get("role") == "ball"
            and (detection.get("bbox_pitch") or {}).get("x_bottom_middle") is not None
        ]
        track = clean_ball_track(ball, trajectory_config)
        report["timeline"] = timeline(predictions, track["frames"], track["points"])
        report["identities"] = {
            str(player["identity"]): {
                "team": player["team"],
                "role": player["role"],
                "jersey": player["jersey"],
            }
            for player in report["players"]
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    size_kb = args.output.stat().st_size / 1024
    print(
        f"{report['title']}: {len(report['players'])} identities, "
        f"{report['frames_with_game_state']} frames with game state "
        f"({report['coverage']:.0%}, {report['coverage_basis']}), ball {report['ball']['kept']}/"
        f"{report['ball']['observed']} kept, {calibration.get('frames_rejected', 0)} frames "
        f"dropped as uncalibrated -> {args.output} ({size_kb:.0f} KB)"
    )
    counts = report.get("event_counts") or {}
    if counts:
        print(
            "events: "
            + ", ".join(f"{name} {value}" for name, value in counts.items())
            + f" | score {report['score']['left']}:{report['score']['right']}"
            f" ({report['score_source']})"
        )


if __name__ == "__main__":
    main()
