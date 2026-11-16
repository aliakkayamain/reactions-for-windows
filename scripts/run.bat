@echo off
rem Webcam Reactions - start (virtual camera for Teams / Zoom / Meet)
cd /d "%~dp0.."
if not exist "venv\Scripts\python.exe" (
    echo Not installed yet. Run scripts\setup.bat first.
    pause
    exit /b 1
)
set PYTHONUTF8=1
"venv\Scripts\python.exe" -m webcam_reactions %*
if errorlevel 1 pause
