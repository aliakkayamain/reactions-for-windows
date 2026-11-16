@echo off
rem Webcam Reactions - one-time setup: virtual environment, packages, models, emoji
setlocal
cd /d "%~dp0.."

echo [1/3] Creating Python virtual environment (venv)...
if not exist "venv\Scripts\python.exe" (
    py -3.12 -m venv venv 2>nul || python -m venv venv
)
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Python 3.12 was not found. Install it from https://www.python.org/downloads/
    echo        and tick "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

echo [2/3] Installing packages (this can take a few minutes)...
"venv\Scripts\python.exe" -m pip install --upgrade pip
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Package installation failed. Check your internet connection and try again.
    pause
    exit /b 1
)

echo [3/3] Downloading models and building animated emoji...
"venv\Scripts\python.exe" tools\setup_assets.py
if errorlevel 1 (
    echo ERROR: Asset setup failed. Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo Setup complete. Start the app with scripts\run.bat
pause
