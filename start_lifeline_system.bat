@echo off
title LifeLine Patient Monitoring System
color 0A

echo.
echo ================================================================
echo                LifeLine Patient Monitoring System
echo ================================================================
echo.

echo [1/4] Installing Python API dependencies...
pip install -r requirements_api.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)

echo.
echo [2/4] Installing Frontend dependencies...
cd frontend
call npm install
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install frontend dependencies
    echo Make sure Node.js and npm are installed
    pause
    exit /b 1
)
cd ..

echo.
echo [3/4] Starting Backend API Server...
start "LifeLine API Server" cmd /k "python api_server.py"

echo Waiting for backend to initialize...
timeout /t 8 /nobreak >nul

echo.
echo [4/4] Starting Frontend Development Server...
cd frontend
start "LifeLine Frontend" cmd /k "npm run dev"
cd ..

echo.
echo ================================================================
echo                    System Started Successfully!
echo ================================================================
echo.
echo Backend API:  http://localhost:5000
echo Frontend:     http://localhost:3000
echo.
echo To start monitoring: Run cv_monitor_with_api.py
echo.
echo Press any key to open the frontend in your browser...
pause >nul

start http://localhost:3000

echo.
echo System is running. Close terminal windows to stop services.
echo Press any key to exit this launcher...
pause >nul