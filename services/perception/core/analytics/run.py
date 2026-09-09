"""Turn a finished perception job into a ledger and a per-player report.

    python -m analytics.run webapp/storage/jobs/<job_id> [--kickoff 0 --period 1]

Writes next to the job: events.json, events.sqlite, player_stats.json,
player_stats.csv.  Re-running is cheap — no model is loaded — so the derivation
can be tuned against a job that took hours to produce in seconds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import ball as ball_mod
from . import possession as poss_mod
from . import stats as stats_mod
from .events import EventBuilder
from .ingest import attacking_directions, load_job


def analyse(job_dir: str, kickoff_offset_s=None, period=None, verbose=True) -> dict:
    started = time.time()
    run = load_job(job_dir)
    directions = attacking_directions(run)

    track = ball_mod.build_track(run.ball, run.fps)
    # the span of the clip, not the frames that happened to hold a person:
    # the ball can be tracked in frames where nobody was detected
    coverage = ball_mod.coverage(track, max(run.meta["frames_total"], 1))
    frames = poss_mod.per_frame(track, run.detections, run.fps)
    runs = poss_mod.runs(frames, run.fps)

    builder = EventBuilder(run, track, frames, runs, directions,
                           kickoff_offset_s=kickoff_offset_s, period=period)
    ledger = builder.build()
    ledger.meta["ball_coverage"] = coverage
    ledger.meta["attacking_directions"] = directions
    ledger.meta["possession_runs"] = len(runs)
    ledger.meta["analysed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    report = stats_mod.build(run, ledger, directions)
    report["meta"]["ball_coverage"] = coverage

    ledger.to_json(os.path.join(job_dir, "events.json"))
    ledger.to_sqlite(os.path.join(job_dir, "events.sqlite"))
    with open(os.path.join(job_dir, "player_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    stats_mod.to_frame(report).to_csv(
        os.path.join(job_dir, "player_stats.csv"), index=False, encoding="utf-8-sig")

    if verbose:
        _summary(run, ledger, report, coverage, time.time() - started)
    return report


def _summary(run, ledger, report, coverage, elapsed):
    from collections import Counter
    types = Counter(e.type for e in ledger.events)
    print("клип: %.0f-%.0f с, %d кадров, %.0f fps" % (
        run.meta["start_s"], run.meta["end_s"],
        run.meta["frames_with_detections"], run.fps))
    print("мяч: %d кадров с траекторией (%d наблюдений, %d интерполяций), покрытие %.0f%%" % (
        coverage["frames"], coverage["observed"], coverage["interpolated"],
        100 * coverage["share"]))
    print("события: " + ", ".join("%s=%d" % kv for kv in types.most_common()))
    for team, totals in report["teams"].items():
        print("  %-5s владение %.0f%% | пасов %d | точность %s" % (
            team, 100 * totals["possession_share"], totals["passes"],
            "—" if totals["pass_accuracy"] is None else "%.0f%%" % (100 * totals["pass_accuracy"])))
    print("готово за %.1f с" % elapsed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir")
    parser.add_argument("--kickoff", type=float, default=None,
                        help="seconds into the source video where the period kicked off")
    parser.add_argument("--period", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if not os.path.isfile(os.path.join(args.job_dir, "predictions.json")):
        print("нет predictions.json в %s" % args.job_dir, file=sys.stderr)
        return 2
    analyse(args.job_dir, args.kickoff, args.period, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
