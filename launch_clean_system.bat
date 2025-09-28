@echo off
title LifeLine System Launcher
color 0A

echo.
echo ================================================================
echo                LifeLine Patient Monitoring System
echo                          Clean Launch
echo ================================================================
echo.

echo [STEP 1] Stopping any existing processes...
taskkill /f /im python3.11.exe >nul 2>&1
taskkill /f /im node.exe >nul 2>&1
echo Done.

echo.
echo [STEP 2] Starting Backend API Server...
cd /d "%~dp0"
start "LifeLine API" cmd /k "python api_server.py"
echo Waiting for API to start...
timeout /t 5 /nobreak >nul

echo.
echo [STEP 3] Testing API Connection...
python quick_api_test.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: API test failed, but continuing...
)

echo.
echo [STEP 4] Starting Frontend...
cd frontend
start "LifeLine Frontend" cmd /k "npm run dev"
echo Waiting for frontend to start...
timeout /t 8 /nobreak >nul

echo.
echo [STEP 5] System Status...
echo ================================================================
echo                    LifeLine System Ready!
echo ================================================================
echo.
echo Backend API:     http://localhost:5000
echo Frontend:        http://localhost:3000 (or 3001 if 3000 is busy)
echo.
echo To start monitoring: python cv_monitor_robust.py
echo.
echo Opening frontend in browser...
timeout /t 2 /nobreak >nul

start http://localhost:3000
timeout /t 2 /nobreak >nul
start http://localhost:3001

echo.
echo System is running! Check the opened browser tabs.
echo Press any key to exit launcher...
pause >nul