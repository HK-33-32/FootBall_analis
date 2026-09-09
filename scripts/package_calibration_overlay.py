"""Package the optional deterministic sampler against the frozen legacy kpts source."""

import argparse
import json
import subprocess
from pathlib import Path

from package_core_overlay import digest, make_patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="football-core:optimized-20260908")
    parser.add_argument("--output", type=Path, default=Path("docker/calibration-overlay"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output exists; choose a new release directory")
    original = subprocess.check_output(
        ["docker", "run", "--rm", "--entrypoint", "python", args.image, "-c",
         "from pathlib import Path; print(Path('/app/engine/kpts.py').read_text(), end='')"],
        text=True, encoding="utf-8",
    )
    needle = (
        "    idx = np.random.choice(len(xs), size=min(len(xs), num_random_points), replace=False)"
    )
    if original.count(needle) != 1:
        raise SystemExit("Expected one legacy circle sampler")
    replacement = (
        "    from fi_calibration_sampling import sample_circle_indices\n"
        "    seed = os.environ.get('FG_CALIB_SEED')\n"
        "    idx = sample_circle_indices(len(xs), num_random_points,\n"
        "                                seed=int(seed) if seed is not None else None)"
    )
    after = original.replace(needle, replacement)
    patch = make_patch("engine/kpts.py", original, after)
    args.output.mkdir(parents=True)
    (args.output / "optimizations.patch").write_text(patch, encoding="utf-8", newline="\n")
    manifest = {"schema_version": "core-overlay.v1", "base_image": args.image,
                "patch_sha256": digest(patch), "files": {
                    "engine/kpts.py": {"before": digest(original), "after": digest(after)}}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
