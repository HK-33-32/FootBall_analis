"""Package one-factor CLIP direct-checkpoint loading over the measured runtime."""

import argparse
import json
import subprocess
from pathlib import Path

from package_core_overlay import digest, make_patch

FILES = (
    "engine/jersey_model/CLIPFinetune.py",
    "engine/inference_soccernetGSR.py",
    "engine/tracklet_attributes.py",
)


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Upstream source drift at {before[:80]!r}")
    return source.replace(before, after)


def transform(relative, source):
    if relative == FILES[0]:
        source = replace_once(
            source,
            "        freeze_clip: bool = True\n",
            "        freeze_clip: bool = True,\n        checkpoint_state=None\n",
        )
        source = replace_once(
            source,
            "        self.clip_model, self.preprocess = clip.load(clip_model_name)",
            """        if checkpoint_state is None:
            self.clip_model, self.preprocess = clip.load(clip_model_name)
        else:
            from clip.model import build_model
            from clip.clip import _transform
            from fi_clip_loading import clip_state_from_checkpoint
            backbone = clip_state_from_checkpoint(checkpoint_state, clip_model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model = build_model(backbone).to(device)
            if device == "cpu":
                self.clip_model.float()
            self.preprocess = _transform(self.clip_model.visual.input_resolution)""",
        )
        source = replace_once(
            source,
            "        self.color_projection = nn.Linear(self.clip_feature_dim, color_embedding_dim)",
            """        self.color_projection = nn.Linear(self.clip_feature_dim, color_embedding_dim)
        if checkpoint_state is not None:
            self.load_state_dict(checkpoint_state, strict=True)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, **kwargs):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        return cls(checkpoint_state=state, **kwargs)""",
        )
        return source
    if relative == FILES[1]:
        return replace_once(
            source,
            "            self.jersey_model = CLIPFinetune()\n"
            "            self.jersey_model.load_state_dict("
            "torch.load(model_dir, weights_only=False))",
            "            self.jersey_model = CLIPFinetune.from_checkpoint(model_dir)",
        )
    if relative == FILES[2]:
        return replace_once(
            source,
            "    jersey_model = CLIPFinetune()\n"
            '    jersey_model.load_state_dict(torch.load(cfg["CLIP"]["MODEL_PATH"], '
            "weights_only=False))",
            '    jersey_model = CLIPFinetune.from_checkpoint(cfg["CLIP"]["MODEL_PATH"])',
        )
    raise ValueError(f"Unsupported patch target: {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="football-core:startup-release-20260908")
    parser.add_argument("--output", type=Path, default=Path("docker/clip-loading-overlay"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Do not overwrite an existing code overlay")
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"],
        text=True,
    ).strip()
    code = (
        "import json; from pathlib import Path; "
        f"print(json.dumps({{name:Path('/app', name).read_text() for name in {FILES!r}}}))"
    )
    sources = json.loads(
        subprocess.check_output(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "python",
                image_id,
                "-c",
                code,
            ],
            text=True,
            encoding="utf-8",
        )
    )
    records, patches = {}, []
    for name, source in sources.items():
        candidate = transform(name, source)
        compile(candidate, name, "exec")
        records[name] = {"before": digest(source), "after": digest(candidate)}
        patches.append(make_patch(name, source, candidate))
    helper = Path(__file__).resolve().parents[1] / "src/football_intelligence/clip_loading.py"
    helper_source = helper.read_text(encoding="utf-8")
    helper_name = "engine/fi_clip_loading.py"
    records[helper_name] = {"before": None, "after": digest(helper_source)}
    patches.append(make_patch(helper_name, None, helper_source))
    patch = "".join(patches)
    args.output.mkdir(parents=True)
    (args.output / "optimizations.patch").write_text(patch, encoding="utf-8", newline="\n")
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "core-overlay.v1",
                "base_image_id": image_id,
                "patch_sha256": digest(patch),
                "files": records,
                "loader_version": "clip-direct-state-v1",
                "supported_backbone": "ViT-L/14",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Packaged {len(records)} checked source files")


if __name__ == "__main__":
    main()
