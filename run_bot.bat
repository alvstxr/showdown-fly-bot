@echo off
REM Fly-connectome Pokémon Showdown agent launcher (Windows).
REM Usage:
REM   run_bot.bat --mode ladder --format gen9ou
REM   run_bot.bat --mode challenge --challenge-user YOUR_NAME --format gen9ou
REM   run_bot.bat --local --mode local_eval --n-battles 5

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

python -m pip install --upgrade pip >nul
python -m pip install -q -r requirements.txt
if errorlevel 1 (
  echo ERROR: Failed to install dependencies from requirements.txt
  exit /b 1
)

if exist ".env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" (
      set "%%A=%%B"
    )
  )
) else (
  echo WARNING: no .env file found. Copy .env.example to .env and add credentials.
)

if "%NEUPRINT_APPLICATION_TOKEN%"=="" if "%NEUPRINT_TOKEN%"=="" (
  echo WARNING: NEUPRINT_APPLICATION_TOKEN is empty — a synthetic connectome will be used.
  echo          Get a token at https://neuprint.janelia.org
)

python run_agent.py %*
set "EXITCODE=%ERRORLEVEL%"
endlocal & exit /b %EXITCODE%
