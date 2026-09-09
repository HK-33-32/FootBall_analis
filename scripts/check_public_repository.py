"""Fail if a publication candidate contains common private or oversized artifacts."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

MAX_BYTES = 10 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".avi", ".ckpt", ".gguf", ".mkv", ".mov", ".mp4", ".pth", ".pt", ".zip"}
PATTERNS = {
    "private key": re.compile(rb"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "OpenAI-style secret": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "Hugging Face token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    "personal Windows path": re.compile(rb"[A-Za-z]:[\\/]Users[\\/][^<>:\"/\\|?*\r\n]+[\\/]"),
}


def candidates() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    )
    return [Path(name.decode()) for name in output.split(b"\0") if name]


def main() -> None:
    failures: list[str] = []
    own_path = Path(__file__).resolve()
    for path in candidates():
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_BYTES:
            failures.append(f"oversized ({size} bytes): {path}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden artifact type: {path}")
        if path.resolve() == own_path:
            continue
        content = path.read_bytes()
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                failures.append(f"{label}: {path}")
    if failures:
        raise SystemExit("Repository publication check failed:\n- " + "\n- ".join(failures))
    print(f"Publication check passed for {len(candidates())} files")


if __name__ == "__main__":
    main()
