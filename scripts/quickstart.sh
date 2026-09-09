#!/usr/bin/env sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"

command -v docker >/dev/null 2>&1 || {
  echo "Docker Engine with Compose is required: https://docs.docker.com/engine/install/" >&2
  exit 1
}
docker compose version >/dev/null

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi
mkdir -p data/runtime data/uploads runs weights
export FI_CONTAINER_UID="$(id -u)"
export FI_CONTAINER_GID="$(id -g)"

docker compose up -d --build api
attempt=0
until curl --fail --silent http://localhost:8080/api/v1/health >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 60 ]; then
    docker compose logs --tail 100 api
    echo "API did not become healthy within 120 seconds" >&2
    exit 1
  fi
  sleep 2
done
echo "Football Intelligence is ready: http://localhost:8080"
