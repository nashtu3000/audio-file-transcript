@echo off
REM Setup script for Audio Transcription with Gemini (Windows)
REM Handles PATH issues automatically

echo ==================================================
echo Audio Transcription Setup (Windows)
echo ==================================================
echo.

REM ---- FIND PYTHON ----
set PYCMD=
py --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYCMD=py
    goto :found_python
)

python --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYCMD=python
    goto :found_python
)

REM Python not on PATH - check common install locations
if exist "%LOCALAPPDATA%\Programs\Python\Python3*\python.exe" (
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do set PYCMD=%%D\python.exe
    goto :found_python
)

for /d %%D in ("%LOCALAPPDATA%\Python\pythoncore-*") do (
    if exist "%%D\python.exe" (
        set PYCMD=%%D\python.exe
        goto :found_python
    )
)

echo [X] Python is not installed.
echo     Download from: https://www.python.org/downloads/
echo     Make sure to check "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:found_python
for /f "tokens=*" %%i in ('%PYCMD% --version 2^>^&1') do set PYVER=%%i
echo [OK] %PYVER% found
echo.

REM ---- GET PYTHON DIRECTORIES AND FIX PATH ----
for /f "tokens=*" %%i in ('%PYCMD% -c "import sys, os; print(os.path.dirname(sys.executable))"') do set PYDIR=%%i
set SCRIPTSDIR=%PYDIR%\Scripts

REM Add Python dir to current session PATH if missing
echo %PATH% | findstr /I /C:"%PYDIR%" >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Adding Python directory to PATH: %PYDIR%
    set "PATH=%PYDIR%;%PATH%"
    set PATHFIXED=1
)

REM Add Scripts dir to current session PATH if missing
echo %PATH% | findstr /I /C:"%SCRIPTSDIR%" >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Adding Python Scripts directory to PATH: %SCRIPTSDIR%
    set "PATH=%SCRIPTSDIR%;%PATH%"
    set PATHFIXED=1
)

REM Permanently add to user PATH if we had to fix anything
if defined PATHFIXED (
    echo.
    echo     Saving to user PATH permanently...
    powershell -Command ^
        "$userPath = [Environment]::GetEnvironmentVariable('PATH', 'User');" ^
        "$pyDir = '%PYDIR%';" ^
        "$scriptsDir = '%SCRIPTSDIR%';" ^
        "if ($userPath -notlike \"*$pyDir*\") { $userPath = \"$pyDir;$userPath\" };" ^
        "if ($userPath -notlike \"*$scriptsDir*\") { $userPath = \"$scriptsDir;$userPath\" };" ^
        "[Environment]::SetEnvironmentVariable('PATH', $userPath, 'User')"
    if %errorlevel% equ 0 (
        echo     [OK] PATH updated permanently
        echo     Note: Restart your terminal after setup for PATH changes to take effect.
    ) else (
        echo     [!] Could not update PATH permanently.
        echo         Manually add these to your system PATH:
        echo           %PYDIR%
        echo           %SCRIPTSDIR%
    )
    echo.
)

REM ---- PIP ----
%PYCMD% -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] pip not found, attempting to install...
    %PYCMD% -m ensurepip --upgrade >nul 2>&1
    %PYCMD% -m pip --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo [X] Could not install pip.
        echo     Try manually: %PYCMD% -m ensurepip --upgrade
        echo.
        pause
        exit /b 1
    )
)

echo [OK] pip found
echo.

REM ---- INSTALL DEPENDENCIES ----
echo Installing Python dependencies...
%PYCMD% -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo [X] Failed to install Python dependencies
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] Python dependencies installed
echo.

REM ---- FFMPEG ----
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] ffmpeg is not installed.
    echo.

    REM Try to auto-install with winget if available
    winget --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo     winget detected. Attempting to install ffmpeg automatically...
        echo.
        winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
        if %errorlevel% equ 0 (
            echo.
            echo     [OK] ffmpeg installed via winget
            echo     Note: Restart your terminal for ffmpeg to be available on PATH.
            echo.
            goto :ffmpeg_done
        ) else (
            echo     [!] winget install failed. Try one of the manual methods below.
            echo.
        )
    )

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
)

echo [OK] ffmpeg found
:ffmpeg_done
echo.

REM ---- API KEY ----
if exist .env (
    findstr /C:"GEMINI_API_KEY" .env >nul 2>&1
    if %errorlevel% equ 0 (
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
echo You can now run:
echo   %PYCMD% transcribe_audio.py
echo.
if defined PATHFIXED (
    echo IMPORTANT: Restart your terminal first so PATH changes take effect.
    echo After restarting you can also use: py transcribe_audio.py
    echo.
)
pause
