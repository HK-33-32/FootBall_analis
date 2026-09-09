"""Deployment checks do not require model imports or a GPU."""

import importlib.util
import json
from pathlib import Path

import pytest


def load(relative):
    path = Path(__file__).resolve().parents[2] / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


package = load("scripts/package_core_overlay.py")
installer = load("docker/apply_core_overlay.py")


def fixture_overlay(tmp_path, before="old\n", after="new\n"):
    root, overlay = tmp_path / "app", tmp_path / "overlay"
    root.mkdir()
    overlay.mkdir()
    path = root / "module.py"
    if before is not None:
        path.write_text(before, encoding="utf-8", newline="\r\n")
    patch = package.make_patch("module.py", before, after)
    (overlay / "optimizations.patch").write_text(patch, encoding="utf-8", newline="\n")
    manifest = {"patch_sha256": package.digest(patch), "files": {
        "module.py": {"before": package.digest(before), "after": package.digest(after)},
    }}
    (overlay / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, overlay, path


@pytest.mark.parametrize("before,after", [
    ("old\n", "new\n"), ("old", "new\n"), ("old\n", "new"),
    (None, "new\n"),
])
def test_overlay_exact_hashes_and_windows_line_endings(tmp_path, before, after):
    root, overlay, path = fixture_overlay(tmp_path, before, after)
    installer.apply_overlay(root, overlay)
    assert path.read_bytes() == after.encode()


def test_mismatched_source_is_not_overwritten(tmp_path):
    root, overlay, path = fixture_overlay(tmp_path)
    path.write_text("user edit", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source mismatch"):
        installer.apply_overlay(root, overlay)
    assert path.read_text() == "user edit"


def test_mismatched_patch_is_not_applied(tmp_path):
    root, overlay, path = fixture_overlay(tmp_path)
    (overlay / "optimizations.patch").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        installer.apply_overlay(root, overlay)
    assert path.read_text() == "old\n"


def test_unsafe_manifest_path_is_rejected(tmp_path):
    root, overlay, _ = fixture_overlay(tmp_path)
    manifest_path = overlay / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = {"../outside.py": {"before": None, "after": None}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unsafe overlay path"):
        installer.apply_overlay(root, overlay)


def test_clip_release_overlay_matches_current_helper_and_all_load_sites():
    root = Path(__file__).resolve().parents[2]
    overlay = root / "docker/clip-loading-overlay"
    manifest = json.loads((overlay / "manifest.json").read_text())
    assert installer.normalized_hash(overlay / "optimizations.patch") == manifest["patch_sha256"]
    assert installer.normalized_hash(root / "src/football_intelligence/clip_loading.py") == (
        manifest["files"]["engine/fi_clip_loading.py"]["after"]
    )
    assert set(manifest["files"]) == {
        "engine/fi_clip_loading.py", "engine/jersey_model/CLIPFinetune.py",
        "engine/inference_soccernetGSR.py", "engine/tracklet_attributes.py",
    }
    patch = (overlay / "optimizations.patch").read_text()
    assert patch.count("+            self.jersey_model = CLIPFinetune.from_checkpoint") == 1
    assert patch.count("+    jersey_model = CLIPFinetune.from_checkpoint") == 1
