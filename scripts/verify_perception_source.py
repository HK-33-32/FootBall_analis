"""Verify the integrated perception source against the accepted runtime image."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

EXPECTED_IMAGE_ID = "sha256:7a4900f1893fb6e71ea294c84beeeaaad5b0c6dcbc1bf7c128b29d4d0a36b4ad"
PORTABLE_CONFIG = "engine/configs/config.yaml"
SOURCE_CONFIG_SHA256 = "8bde0e0a0fed473e98d6f5c4b53f49369de7f57791baeb503ea42fc0a3b09cea"
PORTABLE_CONFIG_SHA256 = "e71313fa479cd2304e37a7009a19b26851d92b6cc3a9c84ba3f24ecddc7ff9df"
FETCH_WEIGHTS = "scripts/fetch_weights.py"
SOURCE_FETCH_WEIGHTS_SHA256 = "1150dea59eddf179d4776cd566b1b3666596b7a13c8720babf2bcfc27e91b813"
INTEGRATED_FETCH_WEIGHTS_SHA256 = "6f62b2e2422b52767a7f13dcdd6d6801728ccf70d5558a2232f19f65bff11fdc"
INFERENCE_MODULE = "engine/inference_soccernetGSR.py"
SOURCE_INFERENCE_SHA256 = "0c7caaf369468e240f85dc4aa1097894b26522adf833469f432438fae7fb8c30"
INTEGRATED_INFERENCE_SHA256 = "a493e090615c008d68b6950fb3665553f18aafe0b4175eaf56494da8c569d10c"
TORCHREID_INIT = "engine/reid/torchreid/__init__.py"
SOURCE_TORCHREID_INIT_SHA256 = "cc5aa2a746eaef1c2710f273a3b67f1a0439fea64068343f8a9e6aa13729133f"
INTEGRATED_TORCHREID_INIT_SHA256 = (
    "7094e41f54fdb8105151051a3fed530da7a991ec474c8abd1d4074e7eb0cc5e8"
)
ROOTS = ("core", "engine", "scripts", "rosters")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def local_inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): digest(path.read_bytes())
        for name in ROOTS
        for path in sorted((root / name).rglob("*"))
        if path.is_file()
    }


def image_inventory(image: str, paths: list[str]) -> dict[str, str | None]:
    program = f"""
import hashlib, json
from pathlib import Path
paths = {paths!r}
result = {{}}
for name in paths:
    path = Path('/app') / name
    result[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
print(json.dumps(result, sort_keys=True))
"""
    result = subprocess.run(
        ["docker", "run", "--rm", "-i", "--network", "none", "--entrypoint", "python", image, "-"],
        input=program,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="football-core:clip-release-20260909")
    parser.add_argument("--root", type=Path, default=Path("services/perception"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    image_id = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True
    ).strip()
    if image_id != EXPECTED_IMAGE_ID:
        raise SystemExit(f"Unexpected source image: {image_id}")
    local = local_inventory(args.root)
    required = {
        "core/api.py", "core/jobs.py", "engine/inference_soccernetGSR.py",
        "engine/fi_calibration_sampling.py", "engine/fi_clip_loading.py",
        "engine/prtreid/data/__init__.py", "engine/reid/torchreid/data/__init__.py",
        "engine/yolox/data/__init__.py",
        "engine/sfr/template/soccernet_template_97.npy", "scripts/entrypoint.sh",
    }
    missing = sorted(required - local.keys())
    if missing:
        raise SystemExit(f"Integrated source is incomplete: {missing}")
    remote = image_inventory(args.image, sorted(local))
    changes = {
        name: {"source": remote[name], "integrated": value}
        for name, value in local.items()
        if remote.get(name) != value
    }
    expected_change = {
        PORTABLE_CONFIG: {
            "source": SOURCE_CONFIG_SHA256,
            "integrated": PORTABLE_CONFIG_SHA256,
        },
        FETCH_WEIGHTS: {
            "source": SOURCE_FETCH_WEIGHTS_SHA256,
            "integrated": INTEGRATED_FETCH_WEIGHTS_SHA256,
        },
        INFERENCE_MODULE: {
            "source": SOURCE_INFERENCE_SHA256,
            "integrated": INTEGRATED_INFERENCE_SHA256,
        },
        TORCHREID_INIT: {
            "source": SOURCE_TORCHREID_INIT_SHA256,
            "integrated": INTEGRATED_TORCHREID_INIT_SHA256,
        },
    }
    if changes != expected_change:
        raise SystemExit(f"Unexpected source drift: {json.dumps(changes, indent=2)}")
    report = {
        "schema_version": "perception-source-provenance.v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "source_image": args.image,
        "source_image_id": image_id,
        "files": len(local),
        "exact_files": len(local) - len(changes),
        "declared_portability_changes": changes,
        "excluded": [
            "model weights and data", "nested Git metadata and bytecode",
            "compiled YOLOX extension binaries (runtime import uses Python fallback)",
            "training documentation and demonstration media",
        ],
        "inventory": local,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps({key: report[key] for key in (
        "source_image_id", "files", "exact_files", "declared_portability_changes"
    )}, indent=2))


if __name__ == "__main__":
    main()
