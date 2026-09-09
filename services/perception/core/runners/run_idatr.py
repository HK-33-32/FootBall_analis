"""IDATR post-processing + result JSON in a single process.

The four IDATR scripts and write_json_file_team each spawn their own Python and
two of them load the ReID network again.  Calling their entry points in one
process saves roughly half a minute of start-up per job.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "IDATR"))

import yaml  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    splits = cfg["DATA_SETS"]

    from rmv_doub_bbox import remove_double_bbox
    from gen_tracklets import generate_tracklets
    from refine_tracklets import refine_tracklets
    from create_court_file import post_process
    from tracklet_attributes import refine_tracklet_attributes
    import write_json_file_team as writer

    steps = [
        ("STEP rmv_doub_bbox", lambda: remove_double_bbox(cfg, splits)),
        ("STEP gen_tracklets", lambda: generate_tracklets(cfg, splits)),
        ("STEP refine_tracklets", lambda: refine_tracklets(cfg, splits)),
        ("STEP tracklet_attributes", lambda: refine_tracklet_attributes(cfg, splits)),
        ("STEP create_court_file", lambda: [post_process(cfg, s) for s in splits]),
        ("STEP write_json", lambda: _write_json(writer, cfg, splits)),
    ]
    if not cfg.get("TRACKLET_ATTRIBUTES", True):
        steps = [s for s in steps if "tracklet_attributes" not in s[0]]
    else:
        # teams now come from tracklet clustering, so the colour-word heuristic
        # inside write_json_file_team must not flip them back
        writer.USE_COLOR_TEAM_OVERRIDE = False
        writer.USE_GK_ANCHORED_SIDES = True
    timings = {}
    for label, fn in steps:
        started = time.perf_counter()
        print(label, flush=True)
        fn()
        timings[label.replace("STEP ", "")] = round(time.perf_counter() - started, 1)
        print(f"{label} done in {time.perf_counter() - started:.1f}s", flush=True)
    # the job keeps only the tail of the log, and the early steps scroll out of
    # it long before anyone reads it -- so the breakdown goes next to the config
    _write_timings(args.config, timings)
    print("STEP timings: %s" % timings, flush=True)
    return 0


def _write_timings(config_path, timings) -> None:
    import json

    try:
        target = os.path.join(os.path.dirname(config_path), "idatr_timings.json")
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(timings, fh, ensure_ascii=False, indent=1)
    except OSError:
        pass


def _write_json(writer, cfg, splits):
    for split in splits:
        folder = os.path.join(cfg["DATA_DIR"], split)
        for clip in sorted(os.listdir(folder)):
            clip_path = os.path.join(folder, clip)
            if not os.path.isdir(clip_path):
                continue
            writer.main(os.path.join(clip_path, "img1"), clip_path, clip)


if __name__ == "__main__":
    raise SystemExit(main())
