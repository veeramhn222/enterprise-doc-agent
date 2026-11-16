#!/usr/bin/env bash
# Works with ANY installed Python: the 'uv' helper downloads Python 3.12 automatically if needed.
cd "$(dirname "$0")"
if [ -n "$VIRTUAL_ENV" ]; then
  echo "[ERROR] A virtual environment is active in this terminal. Run 'deactivate' first, then run ./run.sh again."; exit 1
fi
PY=$(command -v python3 || command -v python)
[ -z "$PY" ] && { echo "[ERROR] No Python found. Install Python from https://www.python.org/downloads/"; exit 1; }

if command -v uv >/dev/null 2>&1; then UV=uv; else
  echo "[1/4] Installing the 'uv' helper (one time)..."
  "$PY" -m pip install --user uv || "$PY" -m pip install uv || { echo "[ERROR] could not install uv"; exit 1; }
  UV="$PY -m uv"
fi

# locate the venv's python (Windows layout vs Linux/macOS layout)
find_venv_py() { if [ -x .venv/Scripts/python.exe ]; then echo .venv/Scripts/python.exe; elif [ -x .venv/bin/python ]; then echo .venv/bin/python; fi; }
VPY=$(find_venv_py)
# discard an old virtualenv built with an unsupported Python (older than 3.10 or newer than 3.13)
if [ -n "$VPY" ] && ! "$VPY" -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)"; then
  echo "[INFO] Existing .venv uses an unsupported Python; recreating it."; rm -rf .venv; VPY=""
fi
if [ -z "$VPY" ]; then
  rm -rf .venv
  echo "[2/4] Creating Python 3.12 environment (downloads Python 3.12 if you do not have it)..."
  $UV venv --python 3.12 .venv || { echo "[ERROR] environment creation failed"; exit 1; }
  VPY=$(find_venv_py)
fi
"$VPY" --version
echo "[3/4] Installing packages (first time: a few minutes, progress shown)..."
$UV pip install --python "$VPY" -r requirements.txt || { echo "[ERROR] package install failed"; exit 1; }
echo "[4/4] Starting server. Open http://localhost:8000 when you see 'Uvicorn running'."
"$VPY" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
