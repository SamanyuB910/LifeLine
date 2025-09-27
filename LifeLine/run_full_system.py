# run_full_system.py - Complete CV Monitor System Launcher
import subprocess
import sys
import os
import time
import webbrowser

print("🚀 Starting CV Monitor System...")
print("=" * 50)

# Set environment variables to reduce Streamlit prompts
os.environ['STREAMLIT_BROWSER_GATHERUSAGESTATS'] = 'false'

# Start CV Monitor in background
print("📹 Starting CV Monitor...")
cv_process = subprocess.Popen([
    sys.executable, "cv_monitor_simple.py"
], creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)

# Wait a moment for CV monitor to initialize
time.sleep(3)

# Start Dashboard
print("🌐 Starting Dashboard...")
dashboard_process = subprocess.Popen([
    sys.executable, "-m", "streamlit", "run", "streamlit_dashboard.py",
    "--server.port", "8507",
    "--server.headless", "true",
    "--browser.gatherUsageStats", "false"
])

# Wait for dashboard to start
time.sleep(5)

# Open browser
print("🔗 Opening dashboard in browser...")
webbrowser.open("http://localhost:8507")

print("\n✅ System Started Successfully!")
print("📹 CV Monitor: Running in separate window")
print("🌐 Dashboard: http://localhost:8507")
print("\nPress Ctrl+C to stop both services")

try:
    # Keep the script running
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Stopping services...")
    cv_process.terminate()
    dashboard_process.terminate()
    print("✅ All services stopped!")
