"""Create, verify or install a portable external model-weight bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "weights-manifest.json"
CHUNK_SIZE = 8 * 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "weights-bundle.v1":
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema_version')}")
    paths = [item["path"] for item in manifest["files"]]
    if len(paths) != len(set(paths)):
        raise ValueError("manifest contains duplicate paths")
    for name in paths:
        value = PurePosixPath(name)
        if value.is_absolute() or ".." in value.parts or value.parts[0] in ("", "."):
            raise ValueError(f"unsafe manifest path: {name}")
    return manifest


def verify(weights_dir: Path, manifest: dict) -> list[str]:
    failures = []
    for item in manifest["files"]:
        path = weights_dir / Path(item["path"])
        if not path.is_file():
            failures.append(f"missing: {item['path']}")
            continue
        if path.stat().st_size != item["size_bytes"]:
            failures.append(f"wrong size: {item['path']}")
            continue
        if file_sha256(path) != item["sha256"]:
            failures.append(f"wrong sha256: {item['path']}")
    return failures


def pack(weights_dir: Path, output: Path, manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    failures = verify(weights_dir, manifest)
    if failures:
        raise SystemExit("cannot create bundle:\n" + "\n".join(failures))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing archive: {output}")
    with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
        archive.add(manifest_path, arcname="weights-manifest.json", recursive=False)
        for item in manifest["files"]:
            archive.add(
                weights_dir / Path(item["path"]),
                arcname=f"weights/{item['path']}",
                recursive=False,
            )
    digest = file_sha256(output)
    sidecar = output.with_name(output.name + ".sha256")
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(f"bundle: {output}")
    print(f"sha256: {digest}")


def _safe_members(archive: tarfile.TarFile, manifest: dict) -> dict[str, tarfile.TarInfo]:
    expected = {f"weights/{item['path']}" for item in manifest["files"]}
    expected.add("weights-manifest.json")
    members = {member.name: member for member in archive.getmembers()}
    unexpected = sorted(set(members) - expected)
    missing = sorted(expected - set(members))
    unsafe = sorted(name for name, member in members.items() if not member.isfile())
    if unexpected or missing or unsafe:
        raise ValueError(
            f"invalid bundle layout; unexpected={unexpected}, missing={missing}, non_files={unsafe}"
        )
    return members


def _obtain(source: str, target_dir: Path) -> tuple[Path, bool]:
    if source.startswith(("https://", "http://")):
        destination = target_dir / "downloaded-weights.tar"
        print(f"downloading {source}")
        with urllib.request.urlopen(source) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output, CHUNK_SIZE)
        return destination, True
    return Path(source).expanduser().resolve(), False


def install(
    source: str,
    weights_dir: Path,
    manifest_path: Path,
    archive_sha256: str | None,
    replace: bool,
) -> None:
    manifest = load_manifest(manifest_path)
    weights_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".weights-install-", dir=weights_dir.parent) as raw:
        staging = Path(raw)
        archive_path, _downloaded = _obtain(source, staging)
        if not archive_path.is_file():
            raise SystemExit(f"bundle not found: {archive_path}")
        if archive_sha256 and file_sha256(archive_path) != archive_sha256.lower():
            raise SystemExit("bundle archive SHA-256 mismatch")
        with tarfile.open(archive_path, "r:") as archive:
            members = _safe_members(archive, manifest)
            embedded = json.load(archive.extractfile(members["weights-manifest.json"]))
            if embedded != manifest:
                raise SystemExit("embedded weight manifest differs from repository manifest")
            extracted = staging / "verified"
            for item in manifest["files"]:
                destination = extracted / Path(item["path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                source_stream = archive.extractfile(members[f"weights/{item['path']}"])
                with destination.open("wb") as output:
                    shutil.copyfileobj(source_stream, output, CHUNK_SIZE)
        failures = verify(extracted, manifest)
        if failures:
            raise SystemExit("bundle verification failed:\n" + "\n".join(failures))
        conflicts = []
        for item in manifest["files"]:
            destination = weights_dir / Path(item["path"])
            if destination.exists() and file_sha256(destination) != item["sha256"]:
                conflicts.append(item["path"])
        if conflicts and not replace:
            raise SystemExit(
                "different files already exist; rerun with --replace:\n" + "\n".join(conflicts)
            )
        for item in manifest["files"]:
            source_path = extracted / Path(item["path"])
            destination = weights_dir / Path(item["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source_path, destination)
    print(f"installed and verified {len(manifest['files'])} model files in {weights_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = parser.add_subparsers(dest="command", required=True)

    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")

    pack_parser = commands.add_parser("pack")
    pack_parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    pack_parser.add_argument("--output", type=Path, required=True)

    install_parser = commands.add_parser("install")
    install_parser.add_argument("source", help="local tar path or HTTP(S) URL")
    install_parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    install_parser.add_argument("--archive-sha256")
    install_parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    if args.command == "verify":
        failures = verify(args.weights_dir.resolve(), load_manifest(manifest))
        if failures:
            raise SystemExit("weight verification failed:\n" + "\n".join(failures))
        print("all model files match the manifest")
    elif args.command == "pack":
        pack(args.weights_dir.resolve(), args.output.resolve(), manifest)
    else:
        install(
            args.source,
            args.weights_dir.resolve(),
            manifest,
            args.archive_sha256,
            args.replace,
        )


if __name__ == "__main__":
    main()
