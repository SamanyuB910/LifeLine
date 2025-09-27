@echo off
echo Starting CV Monitor...
echo.
echo Instructions:
echo - Press 'q' to quit the CV monitor
echo - The dashboard will be available at http://localhost:8501
echo.

start "CV Monitor" python cv_monitor.py
timeout /t 3 /nobreak >nul
start "Dashboard" streamlit run streamlit_dashboard.py

echo Both applications started!
echo Close this window or press Ctrl+C to stop.
pause
