#!/usr/bin/env sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"
command -v docker >/dev/null 2>&1 || {
  echo "Docker Engine with Compose is required" >&2
  exit 1
}
docker compose version >/dev/null
[ -f .env ] || cp .env.example .env
mkdir -p weights/checkpoints data/runtime data/uploads runs
export FI_CONTAINER_UID="$(id -u)"
export FI_CONTAINER_GID="$(id -g)"

echo "Checking Docker GPU access..."
docker run --rm --gpus all --entrypoint nvidia-smi \
  nvidia/cuda:12.8.1-base-ubuntu24.04 >/dev/null
echo "Building the integrated perception service (first build is large)..."
docker compose --profile full build perception
echo "Downloading missing public core checkpoints..."
docker compose --profile full run --rm perception fetch-weights

for name in CLIP_Jersey.pth SoccernetGSR_EfficientNet_Best.pth sports_model.pth.tar-60; do
  if [ ! -f "weights/checkpoints/$name" ]; then
    echo "Missing required checkpoint: weights/checkpoints/$name" >&2
    exit 1
  fi
done

export FI_CORE_URL=http://perception:8000
export FI_PERCEPTION_URL=http://perception:8000
docker compose --profile full up -d --build api perception

attempt=0
until curl --fail --silent http://127.0.0.1:8000/v1/health >/dev/null \
  && curl --fail --silent http://127.0.0.1:8080/api/v1/health >/dev/null; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 120 ]; then
    docker compose logs --tail 150 api perception
    echo "Full stack did not become healthy within 240 seconds" >&2
    exit 1
  fi
  sleep 2
done
echo "Full stack is ready: http://localhost:8080"
