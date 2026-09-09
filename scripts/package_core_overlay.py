"""Freeze the measured external core optimizations as a hash-checked Docker patch."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

FILES = (
    "core/jobs.py",
    "core/runners/run_idatr.py",
    "engine/inference_soccernetGSR.py",
    "engine/IDATR/gen_tracklets.py",
    "engine/tracklet_attributes.py",
    "engine/jersey_batch.py",
    "engine/fast_prep.py",
)


def digest(value: str | None) -> str | None:
    return hashlib.sha256(value.encode()).hexdigest() if value is not None else None


def make_patch(relative: str, original: str | None, after: str) -> str:
    lines = difflib.unified_diff(
        (original or "").splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{relative}" if original is not None else "/dev/null",
        tofile=f"b/{relative}",
    )
    return "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
                   for line in lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base-image", default="football-core:1.0.0")
    parser.add_argument("--output", type=Path, default=Path("docker/core-overlay"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output exists; choose a new directory, do not overwrite a release")
    code = (
        "from pathlib import Path; import json; "
        f"files={FILES!r}; "
        "print(json.dumps({p:Path('/app',p).read_text() "
        "if Path('/app',p).exists() else None for p in files}))"
    )
    before = json.loads(subprocess.check_output(
        ["docker", "run", "--rm", "--entrypoint", "python", args.base_image, "-c", code],
        text=True, encoding="utf-8",
    ))
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", args.base_image], text=True,
    ).strip()
    patch, records = [], {}
    for relative in FILES:
        after = (args.source / relative).read_text(encoding="utf-8")
        original = before[relative]
        records[relative] = {"before": digest(original), "after": digest(after)}
        if original != after:
            patch.append(make_patch(relative, original, after))
    args.output.mkdir(parents=True)
    content = "".join(patch)
    (args.output / "optimizations.patch").write_text(content, encoding="utf-8", newline="\n")
    manifest = {"schema_version": "core-overlay.v1", "base_image": args.base_image,
                "base_image_id": image_id, "patch_sha256": digest(content), "files": records}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n",
    )
    print(json.dumps({"files": len(records), "patch_lines": len(content.splitlines()),
                      "image": image_id}))


if __name__ == "__main__":
    main()
