"""
Integration Test Script
=======================

Test script to verify the integration between the LifeLine backend API
and the Next.js frontend is working correctly.
"""

import requests
import json
import time
import sys

def test_api_endpoints():
    """Test all API endpoints"""
    base_url = "http://localhost:5000"
    
    endpoints_to_test = [
        ("/api/patients/patient_001", "Patient Data"),
        ("/api/alerts", "All Alerts"),
        ("/api/alerts/ongoing", "Ongoing Alerts"),
        ("/api/alerts/resolved", "Resolved Alerts"),
        ("/api/alerts/previous", "Previous Alerts"),
        ("/api/system-status", "System Status"),
        ("/api/video-stream-status", "Video Stream Status")
    ]
    
    print("=" * 60)
    print("LifeLine Integration Test")
    print("=" * 60)
    print(f"Testing API endpoints at: {base_url}")
    print("-" * 60)
    
    all_passed = True
    
    for endpoint, description in endpoints_to_test:
        try:
            print(f"Testing {description:<20} {endpoint:<25} ", end="")
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ PASS")
                
                # Basic data validation
                if endpoint == "/api/patients/patient_001":
                    if 'name' in data and 'room' in data and 'emotion' in data:
                        print(f"    → Patient: {data.get('name')} in {data.get('room')}")
                    else:
                        print("    ⚠️  Missing expected patient data fields")
                
                elif endpoint == "/api/alerts":
                    if 'ongoing' in data and 'resolved' in data and 'previous' in data:
                        print(f"    → Alerts - Ongoing: {len(data['ongoing'])}, Resolved: {len(data['resolved'])}, Previous: {len(data['previous'])}")
                    else:
                        print("    ⚠️  Missing expected alert categories")
                
                elif endpoint == "/api/system-status":
                    status = data.get('status', 'unknown')
                    print(f"    → System Status: {status}")
                    
            else:
                print(f"❌ FAIL (HTTP {response.status_code})")
                all_passed = False
                
        except requests.exceptions.ConnectionError:
            print("❌ FAIL (Connection refused)")
            print("    → Is the API server running at http://localhost:5000?")
            all_passed = False
            
        except requests.exceptions.Timeout:
            print("❌ FAIL (Timeout)")
            all_passed = False
            
        except Exception as e:
            print(f"❌ FAIL ({str(e)})")
            all_passed = False
    
    print("-" * 60)
    
    if all_passed:
        print("🎉 All tests PASSED! Integration is working correctly.")
        print("\nNext steps:")
        print("1. Make sure frontend is running at http://localhost:3000")
        print("2. Start CV monitoring with: python cv_monitor_with_api.py")
        print("3. Open http://localhost:3000 in your browser")
    else:
        print("⚠️  Some tests FAILED. Check the API server.")
        print("\nTroubleshooting:")
        print("1. Make sure API server is running: python api_server.py")
        print("2. Check if ports 5000 and 3000 are available")
        print("3. Install dependencies: pip install -r requirements_api.txt")
    
    print("=" * 60)
    return all_passed

def test_csv_files():
    """Test if CSV files are present and readable"""
    import os
    
    csv_files = [
        ("alerts.csv", "Alerts data"),
        ("emotion_predictions.csv", "Emotion predictions"),
        ("snapshots/predictions.csv", "CV predictions")
    ]
    
    print("\nTesting CSV data files:")
    print("-" * 30)
    
    for file_path, description in csv_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    lines = len(f.readlines())
                print(f"✅ {description:<20} ({lines} lines)")
            except Exception as e:
                print(f"⚠️  {description:<20} (Error reading: {e})")
        else:
            print(f"📝 {description:<20} (File not found - will be created)")

def main():
    """Main test function"""
    print("Starting LifeLine Integration Tests...")
    print("Make sure the API server is running before starting tests.")
    print()
    
    # Wait a moment for user to read
    time.sleep(2)
    
    # Test CSV files first
    test_csv_files()
    
    # Test API endpoints
    success = test_api_endpoints()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()