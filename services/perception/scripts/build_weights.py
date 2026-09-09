"""Разложить веса внутри образа: что положили в контекст — на место, чего нет —
скачать.

Файлы из контекста сборки уже лежат на своих местах — COPY кладёт их прямо
туда, куда надо, потому что перекладывание внутри образа создало бы второй слой
той же величины, а удаление первого место не вернуло бы. Здесь остаётся только
догрузить недостающее.

Запускается во время `docker build`, поэтому сеть может быть, а может и не быть,
и вести себя надо соответственно: отсутствие необязательного файла не должно
ронять сборку, а отсутствие обязательного — должно, и с внятным сообщением.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

TARGET = Path(os.environ.get("FG_CHECKPOINT_DIR", "/opt/weights/checkpoints"))
RF_HOME = Path(os.environ.get("RF_HOME", "/opt/weights/rfdetr"))
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1kgZGxUYGkYhM9AwHzjuSVo7r7FFZ16mh"

REQUIRED = ("SoccernetGSR_EfficientNet_Best.pth", "CLIP_Jersey.pth",
            "sports_model.pth.tar-60")


def download_missing() -> list[str]:
    missing = [n for n in REQUIRED if not (TARGET / n).is_file()]
    if not missing:
        return []
    print("не хватает: %s — качаю из Google Drive" % ", ".join(missing), flush=True)
    try:
        import gdown
        gdown.download_folder(url=DRIVE_FOLDER, output=str(TARGET),
                              quiet=False, use_cookies=False)
    except Exception as exc:
        print("загрузка не удалась: %s" % exc, file=sys.stderr)
    return [n for n in REQUIRED if not (TARGET / n).is_file()]


def prefetch_rfdetr() -> None:
    """Дёрнуть загрузчик RF-DETR, чтобы первый прогон не ждал сети."""
    size = os.environ.get("FG_DEFAULT_DETECTOR_SIZE", "large")
    class_name = {"nano": "RFDETRNano", "small": "RFDETRSmall",
                  "medium": "RFDETRMedium", "base": "RFDETRBase",
                  "large": "RFDETRLarge"}.get(size, "RFDETRLarge")
    if list(RF_HOME.glob("models/*%s*.pth" % size)):
        print("RF-DETR %s уже в образе" % size)
        return
    print("качаю веса RF-DETR (%s)" % size, flush=True)
    try:
        import rfdetr
        getattr(rfdetr, class_name)()      # конструктор скачивает чекпойнт
        print("RF-DETR готов")
    except Exception as exc:
        # не обязателен: если сети нет, он скачается при первом прогоне
        print("не удалось предзагрузить RF-DETR: %s" % exc, file=sys.stderr)


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    RF_HOME.mkdir(parents=True, exist_ok=True)
    print("раскладываю веса в %s" % TARGET, flush=True)
    missing = download_missing()
    prefetch_rfdetr()

    gguf = list(TARGET.glob("*.gguf"))
    print("\nитог:")
    for path in sorted(TARGET.glob("*")):
        if path.is_file():
            print("  %-58s %7.0f МБ" % (path.name[:58], path.stat().st_size / 2 ** 20))
    print("  Qwen GGUF: %s" % ("%d файл(ов)" % len(gguf) if gguf else "нет, номера читает CLIP"))

    if missing:
        print("\nОБЯЗАТЕЛЬНЫЕ ВЕСА ОТСУТСТВУЮТ: %s\n"
              "Положите их в weights/checkpoints/ контекста сборки и повторите."
              % ", ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
