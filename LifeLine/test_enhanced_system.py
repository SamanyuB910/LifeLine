#!/usr/bin/env python3
"""
Enhanced LifeLine System Test Script
Tests all new features including security, fall detection, vital signs, and alerts
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_security_system():
    """Test the secure data manager"""
    print("🔐 Testing Security System...")
    
    try:
        from secure_data_manager import SecureDataManager
        
        # Test encryption/decryption
        manager = SecureDataManager("test_password_123")
        print("✅ Secure data manager initialized")
        
        # Test data encryption
        test_data = {
            'patient_id': 'TEST001',
            'timestamp': datetime.now().isoformat(),
            'emotion': 'happy',
            'confidence': 0.85,
            'pain_scale': 2
        }
        
        # Encrypt data
        encrypted_file = manager.encrypt_data(test_data, 'TEST001')
        print(f"✅ Data encrypted: {encrypted_file}")
        
        # Decrypt data
        decrypted_data = manager.decrypt_data(encrypted_file)
        print(f"✅ Data decrypted: {decrypted_data}")
        
        # Test anonymization
        df = pd.DataFrame([test_data])
        anonymized_data, patient_hash = manager.anonymize_patient_data(df, 'TEST001')
        print(f"✅ Data anonymized: Patient hash = {patient_hash}")
        
        # Test audit report
        audit_report = manager.get_audit_report()
        print(f"✅ Audit report generated: {audit_report['total_access_logs']} logs")
        
        return True
        
    except Exception as e:
        print(f"❌ Security system test failed: {e}")
        return False

def test_fall_detection():
    """Test the fall detection system"""
    print("\n🛡️ Testing Fall Detection System...")
    
    try:
        from fall_detection import FallDetectionSystem
        
        fall_detector = FallDetectionSystem()
        print("✅ Fall detection system initialized")
        
        # Test posture analysis
        bbox = (100, 100, 150, 150)  # Sample bounding box
        headpose = (10, -15, 5)  # Sample head pose
        
        posture_analysis = fall_detector.analyze_posture(bbox, headpose)
        print(f"✅ Posture analysis: {posture_analysis['posture']} (risk: {posture_analysis['risk_score']:.2f})")
        
        # Test bed exit detection
        movement_history = [10, 15, 20, 25, 30]  # Sample movement data
        bed_exit, risk = fall_detector.detect_bed_exit(bbox, movement_history)
        print(f"✅ Bed exit detection: {bed_exit} (risk: {risk:.2f})")
        
        # Test fall risk prediction
        analysis = {
            'posture_risk': 0.3,
            'movement_risk': 0.2,
            'bed_exit_risk': 0.1
        }
        fall_risk = fall_detector.predict_fall_risk(analysis)
        print(f"✅ Fall risk prediction: {fall_risk:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fall detection test failed: {e}")
        return False

def test_vital_signs():
    """Test the vital signs monitoring system"""
    print("\n💓 Testing Vital Signs System...")
    
    try:
        from vital_signs import VitalSignsMonitor
        
        vital_monitor = VitalSignsMonitor()
        print("✅ Vital signs monitor initialized")
        
        # Create sample face region (simulated)
        face_region = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test heart rate estimation
        hr = vital_monitor.estimate_heart_rate(face_region)
        print(f"✅ Heart rate estimation: {hr:.1f} BPM")
        
        # Test blood pressure indicators
        bp_indicators = vital_monitor.detect_blood_pressure_indicators(face_region)
        print(f"✅ Blood pressure indicators: {bp_indicators}")
        
        # Test breathing rate
        br = vital_monitor.estimate_breathing_rate(face_region)
        print(f"✅ Breathing rate estimation: {br:.1f} bpm")
        
        # Test comprehensive vitals
        vitals = vital_monitor.get_comprehensive_vitals(face_region)
        print(f"✅ Comprehensive vitals: HR={vitals['heart_rate']:.1f}, BR={vitals['breathing_rate']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vital signs test failed: {e}")
        return False

def test_alert_manager():
    """Test the enhanced alert management system"""
    print("\n🚨 Testing Alert Management System...")
    
    try:
        from alert_manager import EnhancedAlertManager
        
        alert_manager = EnhancedAlertManager("TEST001")
        print("✅ Alert manager initialized")
        
        # Test alert creation
        test_detections = {
            'emotion': 'pain',
            'pain_scale': 8,
            'fall_risk': True,
            'risk_factors': ['head_tilt', 'rapid_movement'],
            'vital_signs': {
                'heart_rate': 110,
                'breathing_rate': 28
            },
            'risk_score': 0.85,
            'timestamp': datetime.now()
        }
        
        alerts = alert_manager.process_detections(test_detections)
        print(f"✅ Processed detections: {len(alerts)} alerts generated")
        
        # Test alert summary
        summary = alert_manager.get_alert_summary()
        print(f"✅ Alert summary: {summary['total_active']} active alerts")
        
        return True
        
    except Exception as e:
        print(f"❌ Alert management test failed: {e}")
        return False

def test_secure_cv_monitor():
    """Test the secure CV monitor integration"""
    print("\n📹 Testing Secure CV Monitor...")
    
    try:
        from cv_monitor_secure import SecureCVMonitor
        
        # Test initialization
        monitor = SecureCVMonitor("TEST001", "test_password_123")
        print("✅ Secure CV monitor initialized")
        
        # Test patient state
        print(f"✅ Patient state initialized: {monitor.patient_id}")
        
        # Test risk score calculation
        risk_score = monitor.calculate_risk_score()
        print(f"✅ Risk score calculation: {risk_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Secure CV monitor test failed: {e}")
        return False

def test_dashboard():
    """Test the secure dashboard"""
    print("\n🌐 Testing Secure Dashboard...")
    
    try:
        import streamlit
        from secure_dashboard import main
        
        print("✅ Secure dashboard module imported successfully")
        print("✅ Dashboard can be run with: streamlit run secure_dashboard.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False

def run_integration_test():
    """Run a complete integration test"""
    print("\n🔗 Running Integration Test...")
    
    try:
        from secure_data_manager import SecureDataManager
        from fall_detection import FallDetectionSystem
        from vital_signs import VitalSignsMonitor
        from alert_manager import EnhancedAlertManager
        
        # Initialize all systems
        secure_manager = SecureDataManager("integration_test_123")
        fall_detector = FallDetectionSystem()
        vital_monitor = VitalSignsMonitor()
        alert_manager = EnhancedAlertManager("INTEGRATION_TEST")
        
        print("✅ All systems initialized")
        
        # Simulate patient monitoring session
        session_data = {
            'patient_id': 'INTEGRATION_TEST',
            'timestamp': datetime.now().isoformat(),
            'emotion': 'neutral',
            'confidence': 0.8,
            'pain_scale': 3,
            'yaw': 5.0,
            'pitch': -10.0,
            'roll': 2.0,
            'face_x': 200,
            'face_y': 150,
            'face_w': 120,
            'face_h': 140,
            'fall_risk': False,
            'risk_factors': [],
            'vital_signs': {
                'heart_rate': 75,
                'breathing_rate': 16,
                'confidence': 0.9
            },
            'risk_score': 0.3
        }
        
        # Process through all systems
        df = pd.DataFrame([session_data])
        anonymized_data, patient_hash = secure_manager.anonymize_patient_data(df, 'INTEGRATION_TEST')
        encrypted_file = secure_manager.encrypt_data(anonymized_data, patient_hash)
        
        # Generate alerts
        alerts = alert_manager.process_detections(session_data)
        
        print(f"✅ Integration test completed:")
        print(f"   - Data encrypted: {encrypted_file}")
        print(f"   - Patient anonymized: {patient_hash}")
        print(f"   - Alerts generated: {len(alerts)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Enhanced LifeLine System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Security System", test_security_system),
        ("Fall Detection", test_fall_detection),
        ("Vital Signs", test_vital_signs),
        ("Alert Management", test_alert_manager),
        ("Secure CV Monitor", test_secure_cv_monitor),
        ("Dashboard", test_dashboard),
        ("Integration Test", run_integration_test)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! The enhanced system is ready for use.")
        print("\n📋 Next Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run secure monitor: python cv_monitor_secure.py")
        print("3. Run dashboard: streamlit run secure_dashboard.py")
        print("4. Set environment variable: export LIFELINE_MASTER_PASSWORD='your_secure_password'")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
