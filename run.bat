@echo off
REM Launcher for Audio Transcription - uses the virtual environment from setup
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo Virtual environment not found. Run setup.bat first.
    echo.
    pause
    exit /b 1
)

REM Refresh PATH from registry so newly installed tools (e.g. ffmpeg) are found
REM without needing to restart the terminal
for /f "tokens=2*" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%B"
for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USR_PATH=%%B"
if defined SYS_PATH if defined USR_PATH set "PATH=%SYS_PATH%;%USR_PATH%"

"%VENV_PYTHON%" "%~dp0transcribe_audio.py" %*
if %errorlevel% neq 0 pause
