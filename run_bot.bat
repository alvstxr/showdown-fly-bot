@echo off
REM Fly-connectome Pokémon Showdown agent launcher (Windows).
REM Offline by default (selfcheck). Goes online only for ladder / challenge / accept.
REM
REM   run_bot.bat
REM   run_bot.bat --mode ladder --format gen9ou
REM   run_bot.bat --mode challenge --challenge-user YOUR_NAME --format gen9ou
REM   run_bot.bat --mode accept --format gen9ou

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python is not installed or not on PATH. Install Python 3.10+ from https://www.python.org/downloads/ and retry.
  exit /b 1
)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
  echo ERROR: Python 3.10+ is required.
  exit /b 1
)

if exist ".venv\Scripts\python.exe" (
  rem Reuse the existing Windows venv.
) else (
  if exist ".venv" (
    echo Existing .venv is not a Windows environment. Recreating...
    rmdir /s /q ".venv"
  ) else (
    echo Creating virtual environment at %CD%\.venv
  )
  python -m venv .venv
  if errorlevel 1 (
    echo ERROR: Failed to create .venv
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo ERROR: Could not activate .venv\Scripts\activate.bat
  exit /b 1
)

if not exist "requirements.txt" (
  echo ERROR: requirements.txt is missing
  exit /b 1
)

python -c "import poke_env, numpy, scipy, dotenv" >nul 2>&1
if errorlevel 1 (
  echo Installing Python packages into .venv (one-time; needs internet)...
  python -m pip install -q -r requirements.txt
  if errorlevel 1 (
    echo ERROR: Failed to install dependencies from requirements.txt
    exit /b 1
  )
)

if exist ".env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" (
      set "%%A=%%B"
    )
  )
)

python run_agent.py %*
set "EXITCODE=%ERRORLEVEL%"
endlocal & exit /b %EXITCODE%
