"""Собрать веса в контекст сборки, чтобы образ получился самодостаточным.

    python scripts/stage_weights.py --from-config ../SoccernetGSR-main/configs/config.yaml
    python scripts/stage_weights.py --checkpoints <путь> --vlm <путь к папке с gguf>

Кладёт всё найденное в `weights/` рядом с Dockerfile. Дальше `docker build`
запекает их в образ, и пользователю образа скачивать уже нечего.

Того, что не нашлось локально, сборка попробует скачать сама: веса SoccerNet
GSR — из публичной папки Google Drive, RF-DETR — своим загрузчиком. Файлы GGUF
для Qwen не качаются нигде автоматически, поэтому только этот шаг может
потребовать ручного участия.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent
STAGE = CORE / "weights"

WANTED = [
    "SoccernetGSR_EfficientNet_Best.pth",
    "CLIP_Jersey.pth",
    "sports_model.pth.tar-60",
    "osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth",
    "yolox_soccernet.pth.tar",
]


def copy(source: Path, target: Path) -> bool:
    """Копировать, только если файла ещё нет или он отличается размером."""
    if target.exists() and target.stat().st_size == source.stat().st_size:
        print("  = %-56s уже в контексте" % source.name[:56])
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    size = source.stat().st_size / 2 ** 20
    print("  + %-56s %7.0f МБ" % (source.name[:56], size), flush=True)
    shutil.copy2(source, target)
    return True


def stage_checkpoints(source_dir: Path) -> None:
    print("веса GSR из %s" % source_dir)
    if not source_dir.is_dir():
        print("  каталог недоступен — сборка попробует скачать сама")
        return
    for name in WANTED:
        candidate = source_dir / name
        if candidate.is_file():
            copy(candidate, STAGE / "checkpoints" / name)
        else:
            print("  ? %-56s нет локально" % name[:56])
    extra = source_dir / "prtreid"
    if extra.is_dir():
        shutil.copytree(extra, STAGE / "checkpoints" / "prtreid", dirs_exist_ok=True)
        print("  + prtreid/")


def stage_rfdetr(rf_home: Path, sizes: list[str]) -> None:
    print("\nвеса RF-DETR из %s" % rf_home)
    models = rf_home / "models"
    if not models.is_dir():
        print("  кэша нет — сборка скачает нужный размер сама")
        return
    for path in sorted(models.glob("*.pth")):
        # только нужные размеры: все пять весят 1.6 ГБ, а прогон берёт один
        if sizes and not any(s in path.name for s in sizes):
            continue
        copy(path, STAGE / "rfdetr" / "models" / path.name)


def stage_vlm(source: Path | None) -> None:
    print("\nфайлы Qwen (GGUF)")
    if source is None:
        print("  путь не задан — образ соберётся без VLM, номера будет читать CLIP")
        return
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted(source.glob("*.gguf"))
    else:
        print("  %s недоступен" % source)
        return
    if not files:
        print("  в %s нет файлов .gguf" % source)
        return
    for path in files:
        copy(path, STAGE / "checkpoints" / path.name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", default=None, help="каталог с весами GSR")
    ap.add_argument("--from-config", default=None,
                    help="config.yaml движка: из него берутся пути к файлам Qwen")
    ap.add_argument("--vlm", default=None, help="файл .gguf или каталог с ними")
    ap.add_argument("--rf-home", default=os.environ.get("RF_HOME") or str(Path.home() / ".roboflow"))
    ap.add_argument("--rfdetr-sizes", nargs="*", default=["large"],
                    help="какие размеры RF-DETR запекать; пусто = все")
    args = ap.parse_args(argv)

    checkpoints = Path(args.checkpoints).expanduser() if args.checkpoints else None
    vlm = Path(args.vlm).expanduser() if args.vlm else None

    if args.from_config:
        import yaml
        cfg = yaml.safe_load(Path(args.from_config).read_text(encoding="utf-8"))
        if checkpoints is None:
            checkpoints = Path(args.from_config).resolve().parent.parent / "checkpoints"
        if vlm is None:
            model = (cfg.get("JERSEY_VLM") or {}).get("MODEL_PATH")
            if model and Path(model).is_file():
                vlm = Path(model).parent

    if checkpoints is None:
        checkpoints = CORE / "engine" / "checkpoints"

    STAGE.mkdir(parents=True, exist_ok=True)
    stage_checkpoints(checkpoints)
    stage_rfdetr(Path(args.rf_home).expanduser(), args.rfdetr_sizes)
    stage_vlm(vlm)

    total = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file())
    print("\nв контексте сборки: %.1f ГБ" % (total / 2 ** 30))
    print("образ добавит их к базе CUDA и torch (~9 ГБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
