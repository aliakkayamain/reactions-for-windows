@echo off
rem Webcam Reactions - one-time export of the object detector to models\dfine_l_obj365.onnx
rem (phone, bottle, pen, cup, headphones). Needs git and ~1.5 GB of temporary disk space.
rem The export tools live in %USERPROFILE%\.dfine_export (short path: PyTorch has very long file names);
rem delete that folder afterwards if you like - the app only needs the .onnx file.
setlocal
cd /d "%~dp0.."
set EXPORT_DIR=%USERPROFILE%\.dfine_export
set PY=%EXPORT_DIR%\Scripts\python.exe

if exist "models\dfine_l_obj365.onnx" (
    echo models\dfine_l_obj365.onnx already exists. Delete it first to export again.
    pause
    exit /b 0
)

echo [1/4] Creating the export environment in %EXPORT_DIR% ...
if not exist "%PY%" (
    py -3.12 -m venv "%EXPORT_DIR%" 2>nul || python -m venv "%EXPORT_DIR%"
)
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision || goto :fail
"%PY%" -m pip install onnx onnxslim pyyaml scipy loguru faster-coco-eval tensorboard calflops transformers || goto :fail

echo [2/4] Getting the D-FINE source code...
if not exist "%EXPORT_DIR%\D-FINE" (
    git clone --depth 1 https://github.com/Peterande/D-FINE.git "%EXPORT_DIR%\D-FINE" || goto :fail
)

echo [3/4] Downloading the Objects365 weights (~128 MB)...
if not exist "%EXPORT_DIR%\dfine_l_obj365_e25.pth" (
    curl -L -f -o "%EXPORT_DIR%\dfine_l_obj365_e25.pth" https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj365_e25.pth || goto :fail
)

echo [4/4] Exporting to ONNX...
"%PY%" tools\export_detector.py "%EXPORT_DIR%\D-FINE" "%EXPORT_DIR%\dfine_l_obj365_e25.pth" models\dfine_l_obj365.onnx l || goto :fail

echo.
echo Done. Restart the app; press O to toggle object detection.
pause
exit /b 0

:fail
echo ERROR: export failed (see the messages above).
pause
exit /b 1
