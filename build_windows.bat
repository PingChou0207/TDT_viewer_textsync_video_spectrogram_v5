@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo Building TDT_viewer_textsync_video_spectrogram_v5 for Windows x64
echo ============================================================

where py >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python launcher was not found.
    echo Install 64-bit Python 3.12 and select Add Python to PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 goto :failed
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :failed

python -m pip install --upgrade pip wheel
if errorlevel 1 goto :failed
python -m pip install -r requirements-build.txt
if errorlevel 1 goto :failed

set "PYQTGRAPH_QT_LIB=PySide6"
set "PYINSTALLER_CONFIG_DIR=%CD%\.pyinstaller-cache"

python -m PyInstaller --noconfirm --clean TDT_viewer_textsync_video_spectrogram_v5.spec
if errorlevel 1 goto :failed

if not exist "dist\TDT_viewer_textsync_video_spectrogram_v5\TDT_viewer_textsync_video_spectrogram_v5.exe" (
    echo ERROR: Expected EXE was not created.
    goto :failed
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Compress-Archive -Path 'dist\TDT_viewer_textsync_video_spectrogram_v5' -DestinationPath 'dist\TDT_viewer_textsync_video_spectrogram_v5_Windows_x64.zip' -Force"
if errorlevel 1 goto :failed

echo.
echo BUILD COMPLETE
echo EXE: %CD%\dist\TDT_viewer_textsync_video_spectrogram_v5\TDT_viewer_textsync_video_spectrogram_v5.exe
echo ZIP: %CD%\dist\TDT_viewer_textsync_video_spectrogram_v5_Windows_x64.zip
echo.
pause
exit /b 0

:failed
echo.
echo BUILD FAILED. Review the error shown above.
echo.
pause
exit /b 1
