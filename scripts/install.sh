#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
os="$(uname -s)"
# PyTorch publishes CUDA builds only for Linux and Windows, so macOS defaults to the CPU build.
if [[ "$os" == Darwin ]]; then default_mode=--cpu; else default_mode=--cuda; fi
mode="${1:-$default_mode}"
case "$mode" in
  --cuda)
    if [[ "$os" == Darwin ]]; then echo "CUDA is not available on macOS. Run: bash scripts/install.sh --cpu"; exit 2; fi
    torch_index="https://download.pytorch.org/whl/cu128" ;;
  --cpu) torch_index="https://download.pytorch.org/whl/cpu" ;;
  *) echo "Usage: bash scripts/install.sh [--cuda|--cpu]"; exit 2 ;;
esac
# The pinned vtk 9.5 ships wheels only up to Python 3.13; PYTHON overrides the search.
python=""
for candidate in ${PYTHON:-python3 python3.13 python3.12}; do
  if command -v "$candidate" >/dev/null 2>&1 \
      && "$candidate" -c 'import sys; sys.exit(not (3, 12) <= sys.version_info[:2] <= (3, 13))'; then
    python="$candidate"; break
  fi
done
if [[ -z "$python" ]]; then
  echo "Python 3.12 or 3.13 is required: the pinned vtk 9.5 has no wheels for newer versions."
  echo "Install one (macOS: brew install python@3.13) or point to it: PYTHON=/path/to/python3.13 bash scripts/install.sh"
  exit 1
fi
if [[ "$os" == Darwin ]]; then
  arch="$("$python" -c 'import platform; print(platform.machine())')"
  if [[ "$arch" != arm64 ]]; then
    echo "$python runs as $arch, but PyTorch >= 2.8 for macOS exists only for Apple Silicon (arm64)."
    echo "Use a native arm64 Python (not under Rosetta). Intel Macs are not supported."
    exit 1
  fi
fi
echo "Using $python ($("$python" -V 2>&1))"
"$python" -m venv "$project_dir/.venv"
"$project_dir/.venv/bin/python" -m pip install --upgrade pip
"$project_dir/.venv/bin/python" -m pip install 'torch>=2.8,<3' --index-url "$torch_index"
"$project_dir/.venv/bin/python" -m pip install -e "${project_dir}[desktop]"
echo "Ready. Run: bash scripts/start.sh"
