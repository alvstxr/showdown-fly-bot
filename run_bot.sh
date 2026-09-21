#!/usr/bin/env bash
# Fly-connectome Pokémon Showdown agent launcher (Linux / macOS).
# Offline by default (selfcheck). Goes online only for ladder / challenge / accept.
#
#   chmod +x run_bot.sh
#   ./run_bot.sh
#   ./run_bot.sh --mode ladder --format gen9ou
#   ./run_bot.sh --mode challenge --challenge-user YOUR_NAME --format gen9ou
#   ./run_bot.sh --mode accept --format gen9ou

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  die "Python 3 is not installed or not on PATH. Install Python 3.10+ and retry."
fi

"$PY" - <<'PY' || die "Python 3.10+ is required."
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY

if [ ! -x ".venv/bin/python" ] && [ ! -x ".venv/bin/python3" ]; then
  if [ -d ".venv" ]; then
    echo "Existing .venv is not a Unix environment. Recreating..."
    rm -rf .venv
  else
    echo "Creating virtual environment at $ROOT/.venv"
  fi
  "$PY" -m venv .venv || die "Failed to create .venv (install python3-venv / ensure pip is available)."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if [ ! -f "requirements.txt" ]; then
  die "requirements.txt is missing from $ROOT"
fi

if ! python -c "import poke_env, numpy, scipy, dotenv" >/dev/null 2>&1; then
  echo "Installing Python packages into .venv (one-time; needs internet)..."
  python -m pip install -r requirements.txt
fi

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

exec python run_agent.py "$@"
