"""Read jersey numbers for a set of identities. Runs *inside* the perception image.

This file is copied into the Football Core container and executed with the
container's interpreter, because that is where the GGUF weights and the
``llama_cpp`` runtime live. It reads a manifest written by
``scripts/reread_jerseys.py``, asks the backend's own ``QwenJerseyReader``
about each identity's crops, and writes the full tally back.

It deliberately reuses the backend's reader class rather than reimplementing
the prompt, so the only thing that changes between the backend's own pass and
this one is *which crops the model is shown*.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/app/engine")

import cv2  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        default="/opt/weights/checkpoints/Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
    )
    parser.add_argument(
        "--mmproj", default="/opt/weights/checkpoints/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"
    )
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--crops-per-identity", type=int, default=24)
    parser.add_argument("--min-crops", type=int, default=3)
    args = parser.parse_args()

    from jersey_vlm import QwenJerseyReader

    if not QwenJerseyReader.available(args.model, args.mmproj):
        raise SystemExit(f"jersey VLM not available at {args.model}")

    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    root = os.path.dirname(os.path.abspath(args.manifest))

    started = time.perf_counter()
    reader = QwenJerseyReader(
        args.model,
        args.mmproj,
        crops_per_track=args.crops_per_identity,
        min_votes=1,
        group_size=args.group_size,
    )
    print(f"[vlm] loaded in {time.perf_counter() - started:.1f}s", flush=True)

    results = []
    started = time.perf_counter()
    for entry in manifest["identities"]:
        crops = []
        for view in entry["views"][: args.crops_per_identity]:
            image = cv2.imread(os.path.join(root, view["crop"]))
            if image is not None:
                crops.append(image)
        if len(crops) < args.min_crops:
            results.append(
                {"identity": entry["identity"], "crops": len(crops), "tally": {}, "answers": []}
            )
            continue
        tally, answers = reader.tally(crops)
        results.append(
            {
                "identity": entry["identity"],
                "crops": len(crops),
                "questions": len(answers),
                "tally": {str(number): int(count) for number, count in tally.items()},
                "answers": [None if a is None else int(a) for a in answers],
            }
        )
        print(
            f"[vlm] identity {entry['identity']}: {len(crops)} crops -> {answers}",
            flush=True,
        )

    payload = {
        "sequence": manifest.get("sequence"),
        "model": os.path.basename(args.model),
        "group_size": args.group_size,
        "crops_per_identity": args.crops_per_identity,
        "wall_time_s": round(time.perf_counter() - started, 1),
        "identities": results,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[vlm] {len(results)} identities in {payload['wall_time_s']}s -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
