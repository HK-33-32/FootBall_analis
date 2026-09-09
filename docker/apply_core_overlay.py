"""Fail closed if a legacy source or patch differs from the reviewed manifest."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def normalized_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()


def apply_overlay(root: Path, overlay: Path) -> None:
    manifest = json.loads((overlay / "manifest.json").read_text(encoding="utf-8"))
    patch = overlay / "optimizations.patch"
    if normalized_hash(patch) != manifest["patch_sha256"]:
        raise RuntimeError("Core overlay patch checksum mismatch")
    for stage in ("before", "after"):
        for relative, hashes in manifest["files"].items():
            target = (root / relative).resolve()
            if not target.is_relative_to(root.resolve()):
                raise RuntimeError(f"Unsafe overlay path: {relative}")
            if normalized_hash(target) != hashes[stage]:
                raise RuntimeError(f"Core source mismatch ({stage}): {relative}")
        if stage == "before":
            # The legacy Windows-built layer contains CRLF files. The manifest
            # hashes normalized text, and git apply needs the same line endings.
            for relative in manifest["files"]:
                target = root / relative
                if target.exists():
                    target.write_text(target.read_text(encoding="utf-8"),
                                      encoding="utf-8", newline="\n")
            patch.write_text(patch.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
            git = ["git", "-c", "core.autocrlf=false", "apply"]
            subprocess.run([*git, "--check", str(patch.resolve())], cwd=root, check=True)
            subprocess.run([*git, str(patch.resolve())], cwd=root, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=Path("/opt/core-overlay"))
    parser.add_argument("--root", type=Path, default=Path("/app"))
    args = parser.parse_args()
    apply_overlay(args.root, args.overlay)
