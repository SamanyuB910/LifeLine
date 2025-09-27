# run_dashboard.py - Simple script to run the dashboard
import subprocess
import sys
import os

# Set environment variable to skip streamlit welcome
os.environ['STREAMLIT_BROWSER_GATHERUSAGESTATS'] = 'false'

# Run streamlit
try:
    result = subprocess.run([
        sys.executable, "-m", "streamlit", "run", "streamlit_dashboard.py",
        "--server.address", "127.0.0.1",
        "--server.port", "8501",
        "--browser.gatherUsageStats", "false"
    ], check=True)
except KeyboardInterrupt:
    print("Dashboard stopped by user")
except Exception as e:
    print(f"Error running dashboard: {e}")
