"""Checkpoint a real perception job and collect GPU telemetry; restart with --resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx


def save(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, encoding="utf-8").strip()


def collect_artifacts(container: str, job_id: str, folder: Path) -> dict:
    code = (
        "from pathlib import Path; import json; "
        f"root=Path('/data/jobs/{job_id}'); "
        "names=['config.yaml', 'track_timings.json', 'idatr_timings.json']; "
        "print(json.dumps({n:(root/n).read_text() for n in names if (root/n).is_file()}))"
    )
    artifacts = json.loads(command("docker", "exec", container, "python", "-c", code))
    hashes = {}
    for name, value in artifacts.items():
        (folder / name).write_text(value, encoding="utf-8", newline="\n")
        hashes[name] = hashlib.sha256(value.encode()).hexdigest()
    return hashes


def collect_effective_models(container: str, job_id: str) -> dict:
    # .pth.tar-60 is a weight too; suffix-only inventories miss OSNet checkpoints.
    code = """
from pathlib import Path
import hashlib, json, yaml
def paths(value):
    if isinstance(value, dict):
        return [p for child in value.values() for p in paths(child)]
    if isinstance(value, list):
        return [p for child in value for p in paths(child)]
    if isinstance(value, str) and value.startswith('/opt/weights/') and Path(value).is_file():
        return [value]
    return []
result = {}
for name in sorted(set(paths(yaml.safe_load(Path(CONFIG).read_text())))):
    with open(name, 'rb') as stream:
        result[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
print(json.dumps(result))
""".replace("CONFIG", repr(f"/data/jobs/{job_id}/config.yaml"))
    return json.loads(command("docker", "exec", container, "python", "-c", code))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-url", default="http://localhost:8010")
    parser.add_argument("--container", required=True)
    parser.add_argument("--local-media", required=True, type=Path)
    parser.add_argument("--media-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration", type=float, default=7.6)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--dataset", help="Dataset/sequence identifier; defaults to media stem")
    parser.add_argument("--split", choices=("development", "validation"), default="development")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    folder = args.output
    if not args.resume:
        folder.mkdir(parents=True, exist_ok=False)
    manifest_path = folder / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["container"] != args.container or manifest["core_url"] != args.core_url:
            raise SystemExit("Resume container/URL does not match original run")
        image_id = command("docker", "inspect", "--format", "{{.Image}}", args.container)
        if manifest["image_id"] != image_id:
            raise SystemExit("Resume image differs from the original run")
    else:
        inventory = subprocess.check_output(
            ["docker", "exec", "-i", args.container, "python", "-"],
            input=Path(__file__).with_name("core_runtime_manifest.py").read_text(encoding="utf-8"),
            text=True, encoding="utf-8",
        )
        with args.local_media.open("rb") as stream:
            media_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        payload = {
            "start": 0, "end": args.duration, "fps": args.fps, "resolution": "1080",
            "detector": "rfdetr", "detector_size": "large", "jersey_mode": "CLIP",
            "jersey_reader": "qwen-vl", "jersey_stride": 3, "calib_stride": 1,
            "render_video": False, "analytics": False,
        }
        manifest = {
            "run_id": folder.name, "timestamp": datetime.now(UTC).isoformat(),
            "git_commit": command("git", "rev-parse", "HEAD"),
            "dirty_worktree": bool(command("git", "status", "--porcelain")),
            "dataset": args.dataset or args.local_media.stem, "split": args.split,
            "config": payload, "config_hash": hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "schema_versions": {"benchmark": "core-profile.v1"},
            "prompt_versions": "identified by runtime source hashes",
            "seed": None, "seed_note": "Legacy pipeline has no unified deterministic seed",
            "container": args.container, "core_url": args.core_url,
            "image_id": command("docker", "inspect", "--format", "{{.Image}}", args.container),
            "media_sha256": media_hash, "media_path": args.media_path,
            "runtime": json.loads(inventory), "status": "prepared",
        }
        save(manifest_path, manifest)
    with httpx.Client(base_url=args.core_url, timeout=60) as client:
        if "job_id" not in manifest:
            if args.resume:
                raise SystemExit("No saved job ID: inspect server jobs before submitting again")
            res = client.post("/v1/videos/register", data={
                "path": args.media_path, "name": folder.name,
            })
            res.raise_for_status()
            payload = {**manifest["config"], "video_id": res.json()["id"]}
            res = client.post("/v1/jobs", json=payload)
            res.raise_for_status()
            manifest.update(job_id=res.json()["id"], submitted_at=time.time(), status="running")
            save(manifest_path, manifest)
        job_id = manifest["job_id"]
        print(f"job_id={job_id}; checkpoint={manifest_path}", flush=True)
        last_stage = None
        while True:
            res = client.get(f"/v1/jobs/{job_id}")
            res.raise_for_status()
            job = res.json()
            save(folder / "core_job.json", job)
            sample = {"timestamp": time.time(), "stage": job.get("stage"),
                      "progress": job.get("progress")}
            try:
                sample["gpu"] = command(
                    "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                )
                sample["container"] = command(
                    "docker", "stats", "--no-stream", "--format", "{{json .}}", args.container,
                )
            except subprocess.CalledProcessError as exc:
                sample["telemetry_error"] = str(exc)
            with (folder / "telemetry.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(sample) + "\n")
            if job.get("stage") != last_stage:
                print(f"{job.get('stage')}: {job.get('progress')}", flush=True)
                last_stage = job.get("stage")
            if job["status"] in {"done", "error", "canceled"}:
                manifest.update(status=job["status"])
                manifest.setdefault("observed_finished_at", time.time())
                manifest["timings"] = job.get("timings", {})
                save(manifest_path, manifest)
                if job["status"] != "done":
                    raise RuntimeError(f"Job {job_id}: {job.get('error')}; see core_job.json")
                break
            time.sleep(3)
        res = client.get(f"/v1/jobs/{job_id}/predictions")
        res.raise_for_status()
        save(folder / "predictions.json", res.json())
        manifest["predictions_sha256"] = hashlib.sha256(res.content).hexdigest()
        manifest["artifacts"] = collect_artifacts(args.container, job_id, folder)
        manifest["effective_models"] = collect_effective_models(args.container, job_id)
        save(manifest_path, manifest)
        print(f"done: {folder}", flush=True)


if __name__ == "__main__":
    main()
