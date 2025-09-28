"""
Quick API Test Script
====================

Test the API endpoints to make sure they're working correctly
and returning proper data for the frontend.
"""

import requests
import json

def test_api():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing LifeLine API Endpoints")
    print("=" * 50)
    
    # Test patient data
    try:
        response = requests.get(f"{base_url}/api/patients/patient_001", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Patient Data:")
            print(f"   Name: {data.get('name')}")
            print(f"   Room: {data.get('room')}")
            print(f"   Emotion: {data.get('emotion', {}).get('current')} ({data.get('emotion', {}).get('confidence'):.2f})")
            print(f"   Fall Risk: {data.get('vitals', {}).get('fall_risk')}%")
            print(f"   Face Detected: {data.get('emotion', {}).get('timestamp')}")
        else:
            print(f"❌ Patient Data Error: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Patient Data Error: {e}")
    
    print()
    
    # Test alerts
    try:
        response = requests.get(f"{base_url}/api/alerts", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Alerts Data:")
            print(f"   Ongoing: {len(data.get('ongoing', []))}")
            print(f"   Resolved: {len(data.get('resolved', []))}")
            print(f"   Previous: {len(data.get('previous', []))}")
            
            # Show latest ongoing alert if any
            ongoing = data.get('ongoing', [])
            if ongoing:
                latest = ongoing[0]
                print(f"   Latest Alert: {latest.get('title')} - {latest.get('message')}")
        else:
            print(f"❌ Alerts Error: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Alerts Error: {e}")

if __name__ == "__main__":
    test_api()