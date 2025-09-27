#!/usr/bin/env python3
"""Test script to v    # Add some no-face entries to test alert system
    for i in range(3):
        timestamp = datetime.now() - timedelta(seconds=i*30)fy the enhanced CV monitoring system"""

import os
import pandas as pd
from datetime import datetime, timedelta
import cv2
import numpy as np

# Create test directories
os.makedirs("snapshots", exist_ok=True)

def create_test_data():
    """Create sample data to test the enhanced alert categorization dashboard"""
    
    # Sample emotion data with some no-face entries
    test_data = []
    
    # Add some normal emotion detections
    emotions = ['happy', 'sad', 'neutral', 'angry', 'surprised']
    for i in range(15):
        timestamp = datetime.now() - timedelta(seconds=i*10)
        emotion = emotions[i % len(emotions)]
        
        # Enhanced detection with varying confidence
        if emotion == 'happy':
            score = 0.85 + np.random.uniform(-0.1, 0.1)  # Higher confidence for happy
        else:
            score = 0.65 + np.random.uniform(-0.15, 0.15)
        
        test_data.append({
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'emotion': emotion,
            'score': max(0.1, min(0.99, score)),
            'yaw': np.random.uniform(-30, 30),
            'pitch': np.random.uniform(-20, 20),  
            'roll': np.random.uniform(-15, 15),
            'face_x': np.random.randint(100, 400),
            'face_y': np.random.randint(80, 300),
            'face_w': np.random.randint(120, 200),
            'face_h': np.random.randint(140, 220)
        })
    
    # Add some no-face entries to test alert system
    for i in range(3):
        timestamp = datetime.now() - timedelta(seconds=i*30)
        test_data.append({
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'emotion': 'no_face',
            'score': 0.0,
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0,
            'face_x': 0,
            'face_y': 0,
            'face_w': 0,
            'face_h': 0
        })
    
    # Save test predictions
    df = pd.DataFrame(test_data)
    df = df.sort_values('timestamp')
    df.to_csv('snapshots/predictions.csv', index=False)
    print(f"✅ Created {len(test_data)} test predictions")
    
    # Create enhanced test alerts with status categorization
    current_time = datetime.now()
    alert_data = [
        # Ongoing alerts (current issues)
        {
            'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
            'alert_type': 'no_face_detected',
            'message': 'Patient not visible in camera frame',
            'severity': 'high',
            'status': 'ongoing',
            'duration': 0
        },
        {
            'timestamp': (current_time - timedelta(seconds=45)).strftime("%Y-%m-%d %H:%M:%S"),
            'alert_type': 'agitation_detected', 
            'message': 'Patient showing signs of agitation (sustained angry expression)',
            'severity': 'medium',
            'status': 'ongoing',
            'duration': 45
        },
        
        # Recently resolved alerts (resolved in last 2 minutes)
        {
            'timestamp': (current_time - timedelta(seconds=75)).strftime("%Y-%m-%d %H:%M:%S"),
            'alert_type': 'movement_alert',
            'message': 'Sudden movement detected: 45.2px displacement',
            'severity': 'low',
            'status': 'resolved',
            'duration': 12
        },
        {
            'timestamp': (current_time - timedelta(minutes=1, seconds=30)).strftime("%Y-%m-%d %H:%M:%S"),
            'alert_type': 'no_face_detected',
            'message': 'Patient not visible for 3.2 seconds',
            'severity': 'high',
            'status': 'resolved',
            'duration': 8
        },
        
        # Previous alerts (older than 2 minutes)
        {
            'timestamp': (current_time - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
            'alert_type': 'agitation_detected',
            'message': 'Patient showing signs of agitation (consecutive sad emotions)',
            'severity': 'medium',
            'status': 'previous',
            'duration': 25
        },
        {
            'timestamp': (current_time - timedelta(minutes=8)).strftime("%Y-%m-%d %H:%M:%S"),
            'alert_type': 'movement_alert',
            'message': 'Significant head movement detected',
            'severity': 'low',
            'status': 'previous', 
            'duration': 5
        },
        {
            'timestamp': (current_time - timedelta(minutes=12)).strftime("%Y-%m-%d %H:%M:%S"),
            'alert_type': 'no_face_detected',
            'message': 'Patient not visible for 15.1 seconds',
            'severity': 'high',
            'status': 'previous',
            'duration': 15
        }
    ]
    
    alerts_df = pd.DataFrame(alert_data)
    alerts_df.to_csv('snapshots/alerts.csv', index=False)
    print(f"✅ Created {len(alert_data)} categorized test alerts")
    print(f"   • {len([a for a in alert_data if a['status'] == 'ongoing'])} ongoing alerts")
    print(f"   • {len([a for a in alert_data if a['status'] == 'resolved'])} resolved alerts")
    print(f"   • {len([a for a in alert_data if a['status'] == 'previous'])} previous alerts")
    
    # Create a sample image for latest.jpg
    sample_img = np.zeros((480, 640, 3), dtype=np.uint8)
    sample_img[:] = (50, 50, 100)  # Dark background
    
    # Add some text
    cv2.putText(sample_img, "ENHANCED ALERT SYSTEM", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(sample_img, "Testing Ongoing/Previous Categorization", (60, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    cv2.putText(sample_img, "Alert Status Management Active", (90, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(sample_img, "5-sec ongoing | 2-min previous threshold", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
    
    cv2.imwrite('snapshots/latest.jpg', sample_img)
    print("✅ Created enhanced test image")
    
    return len(test_data), len(alert_data)

if __name__ == "__main__":
    print("🧪 Testing Enhanced CV Monitoring System with Alert Categorization")
    print("=" * 70)
    
    predictions, alerts = create_test_data()
    
    print(f"📊 Test data summary:")
    print(f"   • {predictions} predictions (including no-face entries)")
    print(f"   • {alerts} categorized alerts with status tracking")
    print(f"   • Sample image created with enhanced alert system info")
    print()
    print("🚀 Ready to test enhanced dashboard!")
    print("   Run: streamlit run streamlit_dashboard.py")
    print()
    print("✨ New alert categorization features to verify:")
    print("   🔴 Ongoing alerts - Current active issues requiring attention")
    print("   🟢 Resolved alerts - Recently resolved issues (within 2 minutes)")
    print("   📜 Previous alerts - Historical alerts (older than 2 minutes)")
    print("   ⏰ Duration tracking - Real-time duration calculation")
    print("   🔄 Auto-resolution - No-face alerts resolve when face returns")
    print("   📊 Alert statistics - Ongoing/resolved/total counts")
    print()
    print("🎯 Testing scenarios:")
    print("   • No-face detection with 5-second ongoing threshold")
    print("   • Automatic alert resolution when patient returns to frame")
    print("   • Agitation and movement alerts with proper categorization")
    print("   • Time-based status transitions (ongoing → previous)")
    print("   • Enhanced dashboard UI with separate alert panels")
