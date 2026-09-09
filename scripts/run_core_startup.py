"""Start one reproducible, isolated ARG–FRA startup experiment on local Docker."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Unique experiment name; never overwritten")
    parser.add_argument("--image", default="football-core:startup-release-20260908")
    parser.add_argument("--profile-init", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,60}", args.name):
        raise SystemExit("Name must contain only lowercase letters, digits, - and _")
    root = Path(__file__).resolve().parents[1]
    output = root / "data/runtime/startup_20260908" / args.name
    if output.exists():
        raise SystemExit(f"Experiment already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = root / "data/runtime/gpu_20260908/cache"
    required = [root / "runs/gpu_20260908_optimized/config.yaml",
                root / "runs/gpu_20260908_optimized/manifest.json",
                root / "data/perception_benchmark/media/ARGFRA-chunk0000/ARGFRA-chunk0000.mp4",
                *[cache / name for name in ("clip", "torch", "rfdetr")]]
    for path in required:
        if not path.exists():
            raise SystemExit(f"Required local input/cache missing: {path}")
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True,
    ).strip()
    container = f"fi-startup-{args.name}"
    command = ["docker", "run", "-d", "--pull", "never", "--name", container,
               "--gpus", "all", "--network", "none", "--shm-size", "8g"]
    mounts = [(root, "/workspace", True), (root / "data/runtime", "/experiments", False),
              (cache / "clip", "/root/.cache/clip", False),
              (cache / "torch", "/opt/weights/torch", False),
              (cache / "rfdetr", "/opt/weights/rfdetr", False)]
    for source, target, readonly in mounts:
        command += ["--mount", f"type=bind,source={source},target={target}"
                    + (",readonly" if readonly else "")]
    command += ["--entrypoint", "python", image_id, "/workspace/scripts/profile_core_startup.py",
                "--config", "/workspace/runs/gpu_20260908_optimized/config.yaml",
                "--source-manifest", "/workspace/runs/gpu_20260908_optimized/manifest.json",
                "--video", "/workspace/data/perception_benchmark/media/ARGFRA-chunk0000/"
                "ARGFRA-chunk0000.mp4", "--output", f"/experiments/startup_20260908/{args.name}",
                "--image-id", image_id]
    if args.profile_init:
        command.append("--profile-init")
    container_id = subprocess.check_output(command, text=True).strip()
    launch = {"container": container, "container_id": container_id, "image_id": image_id,
              "output": str(output), "command": command}
    folder = output.parent / "launches"
    folder.mkdir(exist_ok=True)
    (folder / f"{args.name}.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
    print(json.dumps(launch, indent=2))
    print(f"Inspect: docker logs --tail 20 {container}")
    print(f"Wait: docker wait {container}; then inspect manifest.json status")


if __name__ == "__main__":
    main()
