@echo off
REM Works with ANY installed Python: the uv helper downloads Python 3.12 automatically if needed.
cd /d %~dp0
if defined VIRTUAL_ENV (
  echo [ERROR] A virtual environment is active in this terminal. Run "deactivate" first, then run run.bat again.
  pause
  exit /b 1
)
set PY=
py -3 --version >nul 2>nul && set PY=py -3
if not defined PY python --version >nul 2>nul && set PY=python
if not defined PY (
  echo [ERROR] No Python found. Install it from https://www.python.org/downloads/
  pause
  exit /b 1
)

uv --version >nul 2>nul
if errorlevel 1 (
  echo [1/4] Installing the uv helper, one time...
  %PY% -m pip install --user uv
  if errorlevel 1 (
    %PY% -m pip install uv
    if errorlevel 1 ( echo [ERROR] could not install uv & pause & exit /b 1 )
  )
  set UV=%PY% -m uv
) else (
  set UV=uv
)

set VPY=.venv\Scripts\python.exe
REM discard an old virtualenv built with an unsupported Python (older than 3.10 or newer than 3.13)
if exist %VPY% (
  %VPY% -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)"
  if errorlevel 1 (
    echo [INFO] Existing .venv uses an unsupported Python; recreating it.
    rmdir /s /q .venv
  )
)
if not exist %VPY% (
  echo [2/4] Creating Python 3.12 environment, downloads Python 3.12 if you do not have it...
  %UV% venv --python 3.12 .venv
  if errorlevel 1 ( echo [ERROR] environment creation failed & pause & exit /b 1 )
)
%VPY% --version
echo [3/4] Installing packages, first time takes a few minutes...
%UV% pip install --python %VPY% -r requirements.txt
if errorlevel 1 ( echo [ERROR] package install failed & pause & exit /b 1 )
echo [4/4] Starting server. Open http://localhost:8000 when you see "Uvicorn running".
%VPY% -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
