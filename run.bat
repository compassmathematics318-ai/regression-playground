@echo off
setlocal
title Regression Playground

REM Always work from the folder containing this batch file.
cd /d "%~dp0"

echo ==============================================
echo   Regression Playground Launcher
echo ==============================================
echo.

REM ------------------------------------------------------------
REM 1. Create the virtual environment if it does not exist.
REM ------------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] No .venv found. Creating one...

    where py >nul 2>&1
    if not errorlevel 1 (
        py -3.11 -m venv .venv
        if errorlevel 1 py -m venv .venv
    ) else (
        python -m venv .venv
    )

    if not exist ".venv\Scripts\python.exe" (
        echo.
        echo [ERROR] Could not create .venv.
        echo Make sure Python is installed and available as "py" or "python".
        pause
        exit /b 1
    )
)

REM ------------------------------------------------------------
REM 2. Activate the project's virtual environment.
REM ------------------------------------------------------------
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo.
    echo [ERROR] Could not activate .venv.
    pause
    exit /b 1
)

REM ------------------------------------------------------------
REM 3. Install dependencies only if one of the required imports
REM    is currently missing. Existing packages are left alone.
REM ------------------------------------------------------------
python -c "import fastapi, uvicorn, torch, numpy" >nul 2>&1
if errorlevel 1 (
    echo [SETUP] One or more Python packages are missing.
    echo [SETUP] Installing from requirements.txt...
    python -m pip install -r requirements.txt

    if errorlevel 1 (
        echo.
        echo [ERROR] Package installation failed.
        pause
        exit /b 1
    )
)

REM ------------------------------------------------------------
REM 4. Start the browser shortly after Uvicorn begins launching.
REM ------------------------------------------------------------
echo [START] Opening http://127.0.0.1:8000 ...
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8000'"

REM ------------------------------------------------------------
REM 5. Run FastAPI from the backend folder.
REM ------------------------------------------------------------
cd /d "%~dp0backend"
echo [START] Starting FastAPI server...
echo [INFO]  Keep this window open while using the website.
echo [INFO]  Press CTRL+C to stop the server.
echo.

python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

echo.
echo [STOPPED] Server has stopped.
pause
endlocal
