@echo off
REM Run the integrated LifeLine Streamlit app
python -m streamlit run integrated_lifeline_app.py --server.port 8501 --server.headless true
pause
