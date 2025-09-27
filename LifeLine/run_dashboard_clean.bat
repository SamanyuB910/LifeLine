@echo off
cd /d "c:\VS Code Projects\Sudarshan Omnicure"
set STREAMLIT_EMAIL=""
.venv\Scripts\streamlit.exe run streamlit_dashboard.py --server.port 8501 --server.headless true
