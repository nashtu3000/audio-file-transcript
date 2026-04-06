@echo off
REM Launcher for Audio Transcription - uses the virtual environment from setup
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo Virtual environment not found. Run setup.bat first.
    echo.
    pause
    exit /b 1
)

"%VENV_PYTHON%" "%~dp0transcribe_audio.py" %*
if %errorlevel% neq 0 pause
