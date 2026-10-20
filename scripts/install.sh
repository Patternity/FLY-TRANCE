#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:---cuda}"
case "$mode" in
  --cuda) torch_index="https://download.pytorch.org/whl/cu128" ;;
  --cpu) torch_index="https://download.pytorch.org/whl/cpu" ;;
  *) echo "Usage: bash scripts/install.sh [--cuda|--cpu]"; exit 2 ;;
esac
python3 -c 'import sys; assert sys.version_info >= (3,12), "Python 3.12 or newer is required"'
python3 -m venv "$project_dir/.venv"
"$project_dir/.venv/bin/python" -m pip install --upgrade pip
"$project_dir/.venv/bin/python" -m pip install 'torch>=2.8,<3' --index-url "$torch_index"
"$project_dir/.venv/bin/python" -m pip install -e "${project_dir}[desktop]"
echo "Ready. Run: bash scripts/start.sh"
