#!/usr/bin/env bash
# Собрать образ ядра. Одна команда, два образа: веса отдельно от кода.
#
#   ./scripts/build.sh              обычная сборка
#   ./scripts/build.sh --no-vlm     без чтения номеров через VLM (образ на 9 ГБ легче)
#   ./scripts/build.sh --with-yolox добавить веса YOLOX (+756 МБ)
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${TAG:-1.0.0}"
WITH_VLM=1
WITH_YOLOX=0
for a in "$@"; do
  case "$a" in
    --no-vlm) WITH_VLM=0 ;;
    --with-yolox) WITH_YOLOX=1 ;;
    *) echo "неизвестный аргумент: $a" >&2; exit 2 ;;
  esac
done

if [ ! -d weights/checkpoints ] || [ -z "$(ls -A weights/checkpoints 2>/dev/null)" ]; then
  echo "нет весов в weights/checkpoints — соберите контекст:" >&2
  echo "  python scripts/stage_weights.py --from-config <репозиторий>/configs/config.yaml" >&2
  exit 1
fi

# Образ весов пересобирается только когда веса изменились: он и держит те
# двенадцать гигабайт, которые не должны ездить при каждой правке кода.
echo "── образ весов ──"
docker build -f Dockerfile.weights -t "football-core-weights:${TAG}" .

echo "── образ ядра ──"
docker build --provenance=false --sbom=false \
  --build-arg "WEIGHTS_IMAGE=football-core-weights:${TAG}" \
  --build-arg "WITH_VLM=${WITH_VLM}" \
  --build-arg "WITH_YOLOX=${WITH_YOLOX}" \
  -t "football-core:${TAG}" .

echo
docker image inspect "football-core:${TAG}" --format '{{.Size}}' \
  | awk '{printf "готово: football-core:'"${TAG}"'  %.1f ГБ\n", $1/1024/1024/1024}'
echo "проверка окружения:  docker run --rm --gpus all football-core:${TAG} check"
