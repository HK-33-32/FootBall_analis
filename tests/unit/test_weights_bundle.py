import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/weights_bundle.py"

spec = importlib.util.spec_from_file_location("weights_bundle", SCRIPT)
weights_bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weights_bundle)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path):
    source = tmp_path / "source"
    content = b"model revision fixture"
    model = source / "checkpoints/model.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(content)
    manifest = {
        "schema_version": "weights-bundle.v1",
        "bundle_version": "test",
        "total_size_bytes": len(content),
        "files": [
            {
                "path": "checkpoints/model.bin",
                "size_bytes": len(content),
                "sha256": _sha(content),
                "purpose": "test",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source, manifest_path, content


def test_weight_bundle_round_trip(tmp_path):
    source, manifest_path, content = _fixture(tmp_path)
    archive = tmp_path / "weights.tar"
    weights_bundle.pack(source, archive, manifest_path)
    expected_archive_hash = weights_bundle.file_sha256(archive)
    assert archive.with_name("weights.tar.sha256").read_text().startswith(expected_archive_hash)

    destination = tmp_path / "installed"
    weights_bundle.install(
        str(archive), destination, manifest_path, expected_archive_hash, replace=False
    )
    assert (destination / "checkpoints/model.bin").read_bytes() == content
    assert weights_bundle.verify(destination, weights_bundle.load_manifest(manifest_path)) == []


def test_weight_bundle_rejects_wrong_archive_hash(tmp_path):
    source, manifest_path, _content = _fixture(tmp_path)
    archive = tmp_path / "weights.tar"
    weights_bundle.pack(source, archive, manifest_path)
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        weights_bundle.install(
            str(archive), tmp_path / "installed", manifest_path, "0" * 64, replace=False
        )


def test_release_weight_manifest_is_self_consistent():
    manifest = weights_bundle.load_manifest(ROOT / "configs/weights-manifest.json")
    assert len(manifest["files"]) == 9
    assert sum(item["size_bytes"] for item in manifest["files"]) == manifest["total_size_bytes"]
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])
