@echo off
setlocal EnableDelayedExpansion
REM Setup script for Audio Transcription with Gemini (Windows)
REM Creates a virtual environment to guarantee packages are always available

echo ==================================================
echo Audio Transcription Setup (Windows)
echo ==================================================
echo.

REM ---- FIND PYTHON ----
set "PYCMD="

py --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYCMD=py"
    goto :found_python
)

python --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYCMD=python"
    goto :found_python
)

python3 --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYCMD=python3"
    goto :found_python
)

REM Python not on PATH - check common install locations
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if exist "%%D\python.exe" (
        set "PYCMD=%%D\python.exe"
    )
)
if defined PYCMD goto :found_python

for /d %%D in ("%LOCALAPPDATA%\Python\pythoncore-*") do (
    if exist "%%D\python.exe" (
        set "PYCMD=%%D\python.exe"
    )
)
if defined PYCMD goto :found_python

for /d %%D in ("%ProgramFiles%\Python3*") do (
    if exist "%%D\python.exe" (
        set "PYCMD=%%D\python.exe"
    )
)
if defined PYCMD goto :found_python

echo [X] Python is not installed.
echo     Download from: https://www.python.org/downloads/
echo     Make sure to check "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:found_python
for /f "tokens=*" %%i in ('!PYCMD! --version 2^>^&1') do set "PYVER=%%i"
echo [OK] !PYVER! found
echo.

REM ---- CREATE VIRTUAL ENVIRONMENT ----
set "VENV_DIR=%~dp0.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

if exist "!VENV_PYTHON!" (
    echo [OK] Virtual environment already exists
) else (
    echo Creating virtual environment...
    !PYCMD! -m venv "!VENV_DIR!"
    if !errorlevel! neq 0 (
        echo [X] Failed to create virtual environment.
        echo     Try manually: !PYCMD! -m venv .venv
        echo.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)
echo.

REM ---- PIP (inside venv) ----
"!VENV_PYTHON!" -m pip --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [!] pip not found in venv, attempting to install...
    "!VENV_PYTHON!" -m ensurepip --upgrade >nul 2>&1
)

"!VENV_PYTHON!" -m pip --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [X] Could not find or install pip in virtual environment.
    echo     Try deleting the .venv folder and running setup again.
    echo.
    pause
    exit /b 1
)

echo [OK] pip found
echo.

REM ---- INSTALL DEPENDENCIES (inside venv) ----
echo Installing Python dependencies...
"!VENV_PYTHON!" -m pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo [X] Failed to install Python dependencies
    echo.
    pause
    exit /b 1
)

REM Verify packages are actually importable
"!VENV_PYTHON!" -c "from google import genai; from pydub import AudioSegment; print('ok')" >nul 2>&1
if !errorlevel! neq 0 (
    echo [X] Dependencies installed but not importable.
    echo     Try deleting the .venv folder and running setup again.
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] Python dependencies installed and verified
echo.

REM ---- FFMPEG ----
ffmpeg -version >nul 2>&1
if !errorlevel! equ 0 goto :ffmpeg_ok

echo [!] ffmpeg is not installed.
echo.

REM Try to auto-install with winget if available
winget --version >nul 2>&1
if !errorlevel! neq 0 goto :ffmpeg_manual

echo     winget detected. Attempting to install ffmpeg automatically...
echo.
winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
if !errorlevel! equ 0 (
    echo.
    echo     [OK] ffmpeg installed via winget
    echo     Note: Restart your terminal for ffmpeg to be available on PATH.
    echo.
    goto :ffmpeg_done
)

echo     [!] winget install failed. Try one of the manual methods below.
echo.

:ffmpeg_manual
echo     Install ffmpeg manually using one of these methods:
echo.
echo     Option A: Using winget (built into Windows 10/11)
echo       winget install ffmpeg
echo.
echo     Option B: Using Chocolatey (https://chocolatey.org/)
echo       choco install ffmpeg
echo.
echo     Option C: Manual download
echo       Download from https://ffmpeg.org/download.html
echo       Extract and add the bin\ folder to your system PATH
echo.
pause
exit /b 1

:ffmpeg_ok
echo [OK] ffmpeg found
:ffmpeg_done
echo.

REM ---- API KEY ----
if exist .env (
    findstr /C:"GEMINI_API_KEY" .env >nul 2>&1
    if !errorlevel! equ 0 (
        echo [OK] API key found in .env file
        echo.
    ) else (
        echo [!] .env file exists but no GEMINI_API_KEY found
        echo     The script will prompt you for your API key on first run
        echo     Get your API key from: https://aistudio.google.com/app/apikey
        echo.
    )
) else (
    echo [!] No .env file found
    echo     The script will prompt you for your API key on first run
    echo     and offer to save it automatically.
    echo     Get your API key from: https://aistudio.google.com/app/apikey
    echo.
)

echo ==================================================
echo Setup Complete!
echo ==================================================
echo.
echo To transcribe audio, use:
echo   run.bat
echo.
pause
endlocal
