"""Скачать веса моделей в FG_CHECKPOINT_DIR.

    python scripts/fetch_weights.py            # скачать недостающее
    python scripts/fetch_weights.py --check    # только проверить, ничего не качать

Три источника ведут себя по-разному, и это важно знать заранее:

* Веса SoccerNet GSR лежат в публичной папке Google Drive автора. Качаются
  автоматически, но зависят от чужой папки: если её закроют или переместят,
  загрузка перестанет работать и файлы придётся класть руками.
* RF-DETR скачивает свои веса сам при первом прогоне — в `RF_HOME`
  (по умолчанию ``~/.roboflow/models``). В контейнере это внутренняя
  файловая система, поэтому переменная указывает на смонтированный том, иначе
  веса будут качаться заново после каждого пересоздания контейнера.
* Модель Qwen для чтения номеров автоматически НЕ качается: это файлы GGUF,
  которые нужно взять самому. Без них ядро работает на CLIP-голове.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import settings  # noqa: E402

DRIVE_FOLDER = "https://drive.google.com/drive/folders/1kgZGxUYGkYhM9AwHzjuSVo7r7FFZ16mh"

# что именно нужно и для чего — чтобы отсутствие говорило само за себя
REQUIRED = {
    "SoccernetGSR_EfficientNet_Best.pth": "калибровка поля",
    "CLIP_Jersey.pth": "роль и номер игрока",
    "sports_model.pth.tar-60": "ReID для трекинга",
}
REQUIRED_SHA256 = {
    "CLIP_Jersey.pth": "c068865c6ff28ff8b2dd7e90d2cfd37b72dfe4e117a3be79eb88a5f98e8afa1e",
    "SoccernetGSR_EfficientNet_Best.pth": (
        "3998b0a734c4c595c896d5491ee2f3e062ac24dc0e08ee2f19a00cbaded00cc9"
    ),
    "sports_model.pth.tar-60": (
        "8d5b2fd8763db34c2aad69810466adf413f0426d9f8119d322227e0e639c5fbd"
    ),
}
OPTIONAL = {
    "yolox_soccernet.pth.tar": "детектор YOLOX (нужен только при detector=yolox)",
    "osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth":
        "запасной ReID",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def report(target: Path) -> list[str]:
    print("каталог весов: %s" % target)
    missing = []
    for group, label in ((REQUIRED, "обязательно"), (OPTIONAL, "необязательно")):
        for name, purpose in group.items():
            path = target / name
            valid = path.is_file()
            if valid and name in REQUIRED_SHA256:
                valid = sha256(path) == REQUIRED_SHA256[name]
            mark = "есть" if valid else ("SHA!" if path.is_file() else "НЕТ ")
            size = " %6.0f МБ" % (path.stat().st_size / 2 ** 20) if path.is_file() else ""
            print("  [%s] %-12s %-46s %s%s" % (mark, label, name[:46], purpose, size))
            if not valid and group is REQUIRED:
                missing.append(name)

    rf_home = Path(os.environ.get("RF_HOME") or (Path.home() / ".roboflow"))
    cached = list((rf_home / "models").glob("*.pth")) if (rf_home / "models").is_dir() else []
    print("\nRF-DETR: %s (%s)" % (
        "%d файл(ов) в кэше" % len(cached) if cached else "скачает сам при первом прогоне",
        rf_home))

    vlm = settings.CHECKPOINT_DIR.glob("*.gguf")
    found = list(vlm)
    print("Qwen для номеров: %s" % (
        "%d файл(ов) GGUF" % len(found) if found
        else "не найдено; номера будет читать CLIP-голова"))
    return missing


def download(target: Path) -> int:
    try:
        import gdown
    except ImportError:
        print("нужен gdown: pip install gdown", file=sys.stderr)
        return 1
    target.mkdir(parents=True, exist_ok=True)
    print("качаю из %s\n" % DRIVE_FOLDER)
    try:
        files = gdown.download_folder(url=DRIVE_FOLDER, output=str(target),
                                      quiet=False, use_cookies=False)
    except Exception as exc:
        print("загрузка не удалась: %s" % exc, file=sys.stderr)
        files = None
    if not files:
        print("\nНичего не скачано. Папка Google Drive принадлежит авторам "
              "SoccerNet GSR: её могли закрыть или переместить.\n"
              "Положите файлы в %s вручную — список выше." % target,
              file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="только проверить")
    ap.add_argument("--target", default=None, help="куда класть (иначе FG_CHECKPOINT_DIR)")
    args = ap.parse_args(argv)

    target = Path(args.target).expanduser() if args.target else settings.CHECKPOINT_DIR
    missing = report(target)

    if args.check:
        return 1 if missing else 0
    if not missing:
        print("\nвсё обязательное на месте")
        return 0

    print("\nне хватает: %s" % ", ".join(missing))
    code = download(target)
    print()
    return 1 if report(target) else code


if __name__ == "__main__":
    raise SystemExit(main())
