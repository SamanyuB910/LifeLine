#!/usr/bin/env python3
"""
Enhanced LifeLine System Startup Script
Automatically starts the secure monitoring system and dashboard
"""

import subprocess
import sys
import os
import time
import webbrowser
import threading
from datetime import datetime

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'opencv-python',
        'streamlit', 
        'pandas',
        'numpy',
        'pillow',
        'matplotlib',
        'cryptography'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def setup_environment():
    """Setup environment variables and directories"""
    print("\n🔧 Setting up environment...")
    
    # Set default master password if not set
    if not os.environ.get('LIFELINE_MASTER_PASSWORD'):
        default_password = "lifeline_secure_2024_dev"
        os.environ['LIFELINE_MASTER_PASSWORD'] = default_password
        print(f"⚠️  Using default password: {default_password}")
        print("   Set LIFELINE_MASTER_PASSWORD environment variable for production!")
    
    # Create necessary directories
    directories = [
        "secure_data",
        "secure_data/encrypted", 
        "secure_data/keys",
        "secure_alerts"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def start_secure_monitor(patient_id="default_patient"):
    """Start the secure CV monitor"""
    print(f"\n📹 Starting Secure CV Monitor for patient: {patient_id}")
    
    try:
        # Start the secure monitor in a separate process
        monitor_process = subprocess.Popen([
            sys.executable, "cv_monitor_secure.py"
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Send patient ID and password
        monitor_input = f"{patient_id}\n{os.environ.get('LIFELINE_MASTER_PASSWORD')}\n"
        monitor_process.stdin.write(monitor_input)
        monitor_process.stdin.flush()
        
        print("✅ Secure CV Monitor started")
        return monitor_process
        
    except Exception as e:
        print(f"❌ Failed to start secure monitor: {e}")
        return None

def start_secure_dashboard():
    """Start the secure dashboard"""
    print("\n🌐 Starting Secure Dashboard...")
    
    try:
        # Start Streamlit dashboard
        dashboard_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "secure_dashboard.py",
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
        
        print("✅ Secure Dashboard started on http://localhost:8501")
        return dashboard_process
        
    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
        return None

def open_dashboard():
    """Open dashboard in browser after a delay"""
    time.sleep(5)  # Wait for dashboard to start
    try:
        webbrowser.open("http://localhost:8501")
        print("🌐 Dashboard opened in browser")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("   Please open http://localhost:8501 manually")

def monitor_processes(monitor_process, dashboard_process):
    """Monitor the running processes"""
    print("\n🔄 Monitoring system processes...")
    print("Press Ctrl+C to stop all services")
    
    try:
        while True:
            # Check if processes are still running
            if monitor_process and monitor_process.poll() is not None:
                print("⚠️  CV Monitor process stopped")
                break
            
            if dashboard_process and dashboard_process.poll() is not None:
                print("⚠️  Dashboard process stopped")
                break
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services...")
        
        # Terminate processes
        if monitor_process:
            monitor_process.terminate()
            print("✅ CV Monitor stopped")
        
        if dashboard_process:
            dashboard_process.terminate()
            print("✅ Dashboard stopped")
        
        print("✅ All services stopped successfully!")

def main():
    """Main startup function"""
    print("🚀 Enhanced LifeLine System Startup")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        return False
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Environment setup failed.")
        return False
    
    # Get patient ID
    patient_id = input("\n👤 Enter Patient ID (or press Enter for 'default_patient'): ").strip()
    if not patient_id:
        patient_id = "default_patient"
    
    print(f"\n🎯 Starting system for patient: {patient_id}")
    
    # Start services
    monitor_process = start_secure_monitor(patient_id)
    dashboard_process = start_secure_dashboard()
    
    if not monitor_process or not dashboard_process:
        print("\n❌ Failed to start required services.")
        return False
    
    # Open dashboard in browser
    browser_thread = threading.Thread(target=open_dashboard)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Display system status
    print("\n" + "="*50)
    print("🎉 Enhanced LifeLine System Started Successfully!")
    print("="*50)
    print(f"📹 CV Monitor: Running for patient {patient_id}")
    print("🌐 Dashboard: http://localhost:8501")
    print("🔐 Security: AES-256 encryption enabled")
    print("🛡️ Fall Detection: Active")
    print("💓 Vital Signs: Monitoring")
    print("🚨 Alerts: Advanced management enabled")
    print("\n📋 Features Available:")
    print("   • HIPAA-compliant data encryption")
    print("   • Advanced fall detection and prevention")
    print("   • Real-time vital signs monitoring")
    print("   • Intelligent alert management")
    print("   • Secure web dashboard")
    print("   • Multi-patient support")
    
    # Monitor processes
    monitor_processes(monitor_process, dashboard_process)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        sys.exit(1)
