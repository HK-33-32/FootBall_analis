"""Draw the predictions on every frame of the clip.

Reuses visualize_prediction_results.visualize_predictions, but shards the
frames over worker processes (it is pure OpenCV/CPU work) and then fills the
frames the JSON has no detections for, so the numbering handed to ffmpeg stays
continuous.
"""
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.getcwd())


def _draw_shard(clip_dir, json_path, shard, num_shards, style="fifa"):
    """Runs in a worker: draw the frames whose index hits this shard."""
    sys.path.insert(0, os.getcwd())
    if style == "boxes":
        import visualize_prediction_results as vpr
    else:
        import visualize_fifa as vpr

    img_dir = os.path.join(clip_dir, "img1")
    staging = os.path.join(clip_dir, "visualization_%d" % shard)

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    predictions = data.get("predictions", [])

    image_ids = sorted({p.get("image_id") for p in predictions})
    mine = set(image_ids[shard::num_shards])
    if not mine:
        return 0

    shard_json = os.path.join(clip_dir, "_shard_%d.json" % shard)
    with open(shard_json, "w", encoding="utf-8") as fh:
        json.dump({"predictions": [p for p in predictions if p.get("image_id") in mine]}, fh)

    vpr.visualize_predictions(img_dir, staging, shard_json)
    os.remove(shard_json)
    return len(mine)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--out-dir", required=True, help="flat folder for annotated frames")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--style", choices=("fifa", "boxes"), default="fifa",
                    help="fifa: ground rings and name plates; boxes: raw detector output")
    args = ap.parse_args()

    clip_dir = args.clip_dir
    clip_name = os.path.basename(clip_dir.rstrip("\\/"))
    img_dir = os.path.join(clip_dir, "img1")
    json_path = os.path.join(clip_dir, f"{clip_name}.json")

    if not os.path.isfile(json_path):
        print(f"[visualize] no prediction json at {json_path}", flush=True)
        return 1

    workers = max(1, args.workers)
    if workers == 1:
        _draw_shard(clip_dir, json_path, 0, 1, args.style)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_draw_shard, clip_dir, json_path, shard, workers,
                                   args.style)
                       for shard in range(workers)]
            for future in futures:
                future.result()

    os.makedirs(args.out_dir, exist_ok=True)
    drawn = {}
    for shard in range(workers):
        root = os.path.join(clip_dir, "visualization_%d" % shard, "Predict_Visualization")
        if not os.path.isdir(root):
            continue
        for sub in os.listdir(root):
            sub_path = os.path.join(root, sub)
            if os.path.isdir(sub_path):
                for fn in os.listdir(sub_path):
                    drawn[fn] = os.path.join(sub_path, fn)

    copied = 0
    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith(".jpg"):
            continue
        src = drawn.get(fn, os.path.join(img_dir, fn))
        shutil.copyfile(src, os.path.join(args.out_dir, fn))
        copied += 1

    for shard in range(workers):
        shutil.rmtree(os.path.join(clip_dir, "visualization_%d" % shard), ignore_errors=True)

    print(f"[visualize] {copied} frames written ({len(drawn)} with detections)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
