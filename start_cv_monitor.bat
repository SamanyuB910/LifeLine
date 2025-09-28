@echo off
title LifeLine CV Monitor
color 0B

echo.
echo ================================================================
echo                 LifeLine CV Patient Monitor
echo ================================================================
echo.
echo Make sure the API server is running first!
echo Backend should be available at: http://localhost:5000
echo.
echo Starting Computer Vision Monitoring...
echo.

python cv_monitor_with_api.py

echo.
echo CV Monitor stopped.
pause