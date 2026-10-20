#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ ! -x .venv/bin/fly-trance ]]; then
  echo "Install first: bash scripts/install.sh --cuda (or --cpu)"
  exit 1
fi
exec .venv/bin/fly-trance "$@"
