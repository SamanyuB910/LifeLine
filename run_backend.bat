@echo off
REM Activate venv if present and start backend server
IF EXIST ".venv\Scripts\Activate.ps1" (
    powershell -ExecutionPolicy Bypass -NoProfile -Command "& '.\.venv\Scripts\Activate.ps1'; python backend\server.py"
) ELSE (
    python backend\server.py
)
