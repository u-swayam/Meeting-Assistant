#!/usr/bin/env bash
# One-command setup for Linux / macOS / WSL / Git-Bash.
#   ./setup.sh
# Creates .venv, installs pinned dependencies, creates .env, checks ffmpeg, runs offline tests.
# Re-running is safe. API keys are never handled here: edit .env afterwards.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

say "[1/5] Looking for Python 3.10-3.12 (3.12 recommended)"
PY=""
for c in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
[ -n "$PY" ] || die "Need Python 3.10, 3.11 or 3.12 (3.12 recommended). Install it (e.g. 'brew install python@3.12' or 'sudo apt install python3.12 python3.12-venv') and re-run."
echo "Using: $($PY --version) ($(command -v $PY))"

say "[2/5] Creating virtual environment (.venv)"
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv || die "Could not create venv. On Debian/Ubuntu: sudo apt install python3-venv"
else
  echo ".venv already exists - reusing"
fi
VPY=".venv/bin/python"

say "[3/5] Installing dependencies (first run downloads PyTorch, this takes a few minutes)"
"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r requirements.txt

say "[4/5] Preparing .env and checking ffmpeg"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example -> open it and fill in your API keys."
else
  echo ".env already exists - left untouched"
fi
mkdir -p outputs uploads
if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
  echo "ffmpeg + ffprobe found"
else
  echo "WARNING: ffmpeg/ffprobe NOT found. Install: 'sudo apt install ffmpeg' (Ubuntu) or 'brew install ffmpeg' (macOS)."
fi

say "[5/5] Verifying install (offline tests, no API keys needed)"
"$VPY" -c "import torch, pyannote.audio, openai; print('torch', torch.__version__, '| pyannote', pyannote.audio.__version__)"
"$VPY" -m pytest -q

cat <<MSG

Setup complete.

Next:
  1. Edit .env  (DEEPGRAM_API_KEY, HF_TOKEN, DEEPSEEK_API_KEY)
  2. source .venv/bin/activate
  3. python -m app.server        # then open http://127.0.0.1:8000

MSG
