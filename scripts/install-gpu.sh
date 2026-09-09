#!/usr/bin/env sh
set -eu

if [ "$#" -lt 1 ]; then
  echo "usage: sh scripts/install-gpu.sh <weights.tar-or-url> [sha256]" >&2
  exit 2
fi
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"
command -v python3 >/dev/null 2>&1 || {
  echo "Python 3 is required to verify and install an offline weight bundle" >&2
  exit 1
}
if [ "$#" -ge 2 ]; then
  python3 scripts/weights_bundle.py install "$1" --archive-sha256 "$2"
else
  python3 scripts/weights_bundle.py install "$1"
fi
exec sh scripts/quickstart-gpu.sh
