#!/usr/bin/env bash
# Fail loudly at startup rather than silently at the first job.
set -euo pipefail

check() {
  python - <<'PY'
import shutil, sys
sys.path.insert(0, "/app")
from core import settings

problems = []
if not shutil.which("ffmpeg"):
    problems.append("ffmpeg отсутствует")
try:
    import torch
    if not torch.cuda.is_available():
        problems.append("CUDA недоступна: прогон будет на CPU и займёт в десятки раз дольше")
    else:
        print("GPU: %s" % torch.cuda.get_device_name(0))
except Exception as exc:
    problems.append("torch не импортируется: %s" % exc)

missing = [n for n in ("SoccernetGSR_EfficientNet_Best.pth", "CLIP_Jersey.pth",
                       "sports_model.pth.tar-60")
           if not (settings.CHECKPOINT_DIR / n).is_file()]
if missing:
    problems.append("нет весов в %s: %s" % (settings.CHECKPOINT_DIR, ", ".join(missing)))

try:
    import llama_cpp  # noqa: F401
    print("VLM для номеров: доступен")
except Exception:
    print("VLM для номеров: не собран, номера читает CLIP-голова")

for line in problems:
    print("ВНИМАНИЕ: %s" % line, file=sys.stderr)
print("хранилище: %s" % settings.STORAGE_DIR)
PY
}

case "${1:-serve}" in
  serve)
    check || true
    exec python -m uvicorn core.api:app \
        --host "${FG_HOST:-0.0.0.0}" --port "${FG_PORT:-8000}" \
        --workers 1 --timeout-keep-alive 75
    ;;
  check)
    check
    ;;
  fetch-weights)
    exec python /app/scripts/fetch_weights.py "${@:2}"
    ;;
  shell)
    exec /bin/bash
    ;;
  *)
    exec "$@"
    ;;
esac
