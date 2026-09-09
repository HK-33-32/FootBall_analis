"""Real CLIP loading/equivalence benchmark inside the pinned perception image."""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="/opt/weights/checkpoints/CLIP_Jersey.pth")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "benchmark.py")
    source = json.loads((args.source_run / "manifest.json").read_text())
    if source["split"] != "development" or source["status"] != "complete":
        raise SystemExit("Requires a completed development run, not held-out labels")
    report = {
        "run_id": args.output.name,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": source["git_commit"],
        "dirty_worktree": True,
        "dataset": source["dataset"],
        "split": "development",
        "mode": args.mode,
        "image_id": args.image_id,
        "seed": None,
        "prompt_versions": {},
        "schema_versions": {"clip_loading": "1.0.0"},
        "source_manifest_sha256": digest(args.source_run / "manifest.json"),
        "script_sha256": digest(__file__),
        "status": "initializing",
        "config": {
            "frames": list(range(1, 191, 25)),
            "players_per_frame": 2,
            "input_preprocess": "unchanged CLIP preprocess",
            "device": "cuda",
            "text_probes": ["red", "blue", "white"],
        },
    }
    report["config_hash"] = hashlib.sha256(
        json.dumps(report["config"], sort_keys=True).encode()
    ).hexdigest()
    path = args.output / "manifest.json"
    save(path, report)
    try:
        os.chdir("/app/engine")
        sys.path.insert(0, "/app/engine")
        started = time.perf_counter()
        import cv2
        import torch
        from jersey_model.CLIPFinetune import CLIPFinetune
        from PIL import Image

        report["import_s"] = time.perf_counter() - started
        if not torch.cuda.is_available():
            raise RuntimeError("CLIP equivalence benchmark requires CUDA")
        report["software"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        report["hardware"] = {
            "gpu": torch.cuda.get_device_name(0),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        }
        torch.cuda.synchronize()
        started = time.perf_counter()
        if args.mode == "baseline":
            model = CLIPFinetune()
            model.load_state_dict(torch.load(args.checkpoint, weights_only=False))
        else:
            model = CLIPFinetune.from_checkpoint(args.checkpoint)
        model = model.to("cuda").eval()
        torch.cuda.synchronize()
        report["load_s"] = time.perf_counter() - started
        report["status"] = "probing"
        save(path, report)
        print(f"[clip] mode={args.mode} load_s={report['load_s']:.3f}", flush=True)

        def tensor_record(tensor):
            cpu = tensor.detach().cpu().contiguous()
            raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }

        clip = args.source_run / "data/test/ARGFRA-startup"
        records, counts, images = [], {}, []
        with (clip / "interpolate_ARGFRA-startup.txt").open() as stream:
            for row in csv.reader(stream):
                frame = int(row[0])
                if (
                    frame not in report["config"]["frames"]
                    or row[7] != "Player"
                    or counts.get(frame, 0) >= report["config"]["players_per_frame"]
                ):
                    continue
                filename = clip / "img1" / f"{frame:06d}.jpg"
                image = cv2.imread(str(filename))
                if image is None:
                    raise ValueError(f"Missing source frame: {filename}")
                x, y, width, height = map(float, row[2:6])
                left, top = max(0, int(x)), max(0, int(y))
                right, bottom = (
                    min(image.shape[1], int(x + width)),
                    min(image.shape[0], int(y + height)),
                )
                crop = image[top:bottom, left:right, ::-1].copy()
                if crop.size == 0:
                    raise ValueError("Empty real observation crop")
                value = model.preprocess(Image.fromarray(crop))
                images.append(value)
                records.append(
                    {
                        "frame": frame,
                        "tracklet_id": row[1],
                        "bbox": row[2:6],
                        "frame_sha256": digest(filename),
                        "tensor": tensor_record(value),
                    }
                )
                counts[frame] = counts.get(frame, 0) + 1
        if not images:
            raise ValueError("No real observation crops found")
        batch = torch.stack(images).to("cuda")
        with torch.inference_mode():
            outputs = model(batch)
            for color in report["config"]["text_probes"]:
                outputs[f"text_{color}"] = model.encode_color_text(color)
        torch.cuda.synchronize()
        torch.save(
            {name: tensor.cpu() for name, tensor in outputs.items()}, args.output / "outputs.pt"
        )
        report["inputs"] = records
        report["outputs"] = {name: tensor_record(value) for name, value in outputs.items()}
        report["state"] = {name: tensor_record(value) for name, value in model.state_dict().items()}
        report["requires_grad"] = {
            name: value.requires_grad for name, value in model.named_parameters()
        }
        report["training_flags"] = {name: value.training for name, value in model.named_modules()}
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        report["checkpoint"] = {
            "sha256": digest(args.checkpoint),
            "keys": len(checkpoint),
            "model_keys": len(report["state"]),
            "missing": sorted(set(report["state"]) - set(checkpoint)),
            "unexpected": sorted(set(checkpoint) - set(report["state"])),
            "dtypes": sorted({str(value.dtype) for value in checkpoint.values()}),
        }
        report["models"] = {args.checkpoint: report["checkpoint"]["sha256"]}
        if args.mode == "baseline":
            base = Path("/root/.cache/clip/ViT-L-14.pt")
            report["models"][str(base)] = digest(base)
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["source_hashes"] = {
            str(p): digest(p)
            for p in (
                Path("jersey_model/CLIPFinetune.py"),
                Path("/opt/venv/lib/python3.12/site-packages/clip/model.py"),
                Path("/opt/venv/lib/python3.12/site-packages/clip/clip.py"),
            )
        }
        report["status"] = "complete"
        save(path, report)
        print(
            json.dumps(
                {
                    "state_keys": len(report["state"]),
                    "crops": len(images),
                    "checkpoint": report["checkpoint"],
                }
            ),
            flush=True,
        )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(path, report)
        raise


if __name__ == "__main__":
    main()
