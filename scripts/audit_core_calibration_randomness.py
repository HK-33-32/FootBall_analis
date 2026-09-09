"""Diagnostic simulation: repeat the actual legacy circle sampler, without model inference."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    content = args.source.read_bytes()
    tree = ast.parse(content.decode("utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "extract_circle_points_from_heatmap")
    namespace = {"np": np, "cv2": cv2}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(args.source), "exec"), namespace)
    # Explicit simulation, never production detections or benchmark ground truth.
    y, x = np.mgrid[:384, :384]
    radius = np.sqrt(((x - 192) / 110) ** 2 + ((y - 192) / 70) ** 2)
    heatmap = np.exp(-((radius - 1) / .02) ** 2).astype(np.float32)
    results = {}
    for mode in ("legacy_uncontrolled_stream", "diagnostic_reset_seed_per_call"):
        hashes, counts = [], []
        np.random.seed(0)
        for _ in range(20):
            if mode == "diagnostic_reset_seed_per_call":
                np.random.seed(0)
            points = namespace[function.name](heatmap, num_random_points=100)
            hashes.append(hashlib.sha256(points.tobytes()).hexdigest())
            counts.append(len(points))
        results[mode] = {"unique_outputs": len(set(hashes)), "point_counts": counts,
                         "output_hashes": hashes}
    document = {"type": "explicit_diagnostic_simulation", "seed": 0, "repeats": 20,
                "source_sha256": hashlib.sha256(content).hexdigest(),
                "numpy": np.__version__, "opencv": cv2.__version__, "results": results,
                "limitation": "Not a match accuracy test; no production seed behavior changed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps({name: value["unique_outputs"] for name, value in results.items()}))


if __name__ == "__main__":
    main()
