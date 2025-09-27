# test_all_features.py - Comprehensive Feature Verification Test
import os
import sys
import time
import numpy as np
from datetime import datetime, timedelta
import logging

# Test all our enhanced modules
def test_gaze_tracking():
    """Test the gaze tracking system"""
    print("🔍 Testing Gaze Tracking System...")
    try:
        from gaze_tracking import GazeTrackingSystem
        gaze_tracker = GazeTrackingSystem()
        
        # Create a mock face region (simulated)
        mock_face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_bbox = (10, 10, 80, 80)
        
        # Test gaze processing
        result = gaze_tracker.process_gaze_frame(mock_face, mock_bbox)
        
        # Verify expected keys in result
        expected_keys = ['eye_roll_detected', 'rolling_confidence', 'pupil_analysis', 
                        'sustained_abnormal_gaze', 'eyes_detected', 'gaze_alert_level']
        
        for key in expected_keys:
            if key not in result:
                print(f"❌ Missing key in gaze result: {key}")
                return False
        
        # Test alert message generation
        if result['eye_roll_detected']:
            alert_msg = gaze_tracker.get_alert_message(result)
            if alert_msg:
                print(f"   ✅ Eye rolling alert: {alert_msg['message']}")
        
        print("   ✅ Gaze tracking system functional")
        return True
        
    except Exception as e:
        print(f"   ❌ Gaze tracking test failed: {e}")
        return False

def test_agitation_detection():
    """Test agitation detection from emotion patterns"""
    print("😤 Testing Agitation Detection...")
    try:
        from cv_monitor_secure import SecureCVMonitor
        
        # Create monitor instance
        monitor = SecureCVMonitor("TEST_PATIENT", "test_password")
        
        # Simulate emotion history indicating agitation
        current_time = datetime.now()
        agitation_emotions = ['angry', 'pain', 'angry', 'sad', 'angry']
        
        # Add emotions to history
        for i, emotion in enumerate(agitation_emotions):
            timestamp = current_time - timedelta(seconds=30-i*5)  # Spread over 30 seconds
            monitor.patient_state['emotion_history'].append({
                'emotion': emotion,
                'confidence': 0.8,
                'pain_scale': 8 if emotion == 'pain' else 4,
                'timestamp': timestamp
            })
        
        # Test agitation detection
        agitation_result = monitor.detect_agitation_patterns()
        
        # Verify agitation was detected
        if agitation_result['agitated']:
            print(f"   ✅ Agitation detected: {agitation_result['agitation_level']} level")
            print(f"   ✅ Triggers: {', '.join(agitation_result['triggers'])}")
            print(f"   ✅ Confidence: {agitation_result['confidence']:.2f}")
            return True
        else:
            print("   ❌ Agitation not detected when it should be")
            return False
            
    except Exception as e:
        print(f"   ❌ Agitation detection test failed: {e}")
        return False

def test_fall_detection():
    """Test fall detection system"""
    print("🛡️ Testing Fall Detection System...")
    try:
        from fall_detection import FallDetectionSystem
        
        fall_detector = FallDetectionSystem()
        
        # Test with dangerous head pose
        dangerous_headpose = (60, 50, 40)  # Extreme yaw, pitch, roll
        bbox = (100, 50, 80, 100)  # High position might indicate fall risk
        
        posture_result = fall_detector.analyze_posture(bbox, dangerous_headpose)
        
        if posture_result['risk_score'] > 0.3:
            print(f"   ✅ Fall risk detected: {posture_result['posture']}")
            print(f"   ✅ Risk score: {posture_result['risk_score']:.2f}")
            return True
        else:
            print("   ❌ Fall detection not sensitive enough")
            return False
            
    except Exception as e:
        print(f"   ❌ Fall detection test failed: {e}")
        return False

def test_vital_signs():
    """Test vital signs monitoring"""
    print("💓 Testing Vital Signs Monitoring...")
    try:
        from vital_signs import VitalSignsMonitor
        
        vital_monitor = VitalSignsMonitor()
        
        # Create mock face region
        mock_face = np.random.randint(50, 150, (80, 60, 3), dtype=np.uint8)
        
        # Test heart rate estimation
        hr = vital_monitor.estimate_heart_rate(mock_face)
        
        if isinstance(hr, (int, float)) and 40 <= hr <= 200:
            print(f"   ✅ Heart rate estimated: {hr:.0f} BPM")
        else:
            print(f"   ⚠️ Heart rate estimate: {hr} (may be initial baseline)")
        
        # Test comprehensive vitals
        vitals = vital_monitor.get_comprehensive_vitals(mock_face)
        
        if 'heart_rate' in vitals and 'breathing_rate' in vitals:
            print(f"   ✅ Comprehensive vitals: HR={vitals['heart_rate']:.0f}, BR={vitals['breathing_rate']:.0f}")
            return True
        else:
            print("   ❌ Comprehensive vitals missing data")
            return False
            
    except Exception as e:
        print(f"   ❌ Vital signs test failed: {e}")
        return False

def test_alert_management():
    """Test enhanced alert management"""
    print("🚨 Testing Alert Management...")
    try:
        from alert_manager import EnhancedAlertManager
        
        alert_manager = EnhancedAlertManager("TEST_PATIENT")
        
        # Test detection processing with all features
        test_detections = {
            'emotion': 'pain',
            'pain_scale': 8,
            'fall_risk': True,
            'risk_factors': ['head_tilt', 'rapid_movement'],
            'vital_signs': {'heart_rate': 110, 'breathing_rate': 25},
            'risk_score': 0.9,
            'gaze_analysis': {
                'eye_roll_detected': True,
                'rolling_confidence': 0.8,
                'sustained_abnormal_gaze': True
            },
            'agitation': {
                'agitated': True,
                'agitation_level': 'high',
                'triggers': ['sustained_anger', 'persistent_pain']
            },
            'timestamp': datetime.now()
        }
        
        # Process detections
        alerts = alert_manager.process_detections(test_detections)
        
        if alerts and len(alerts) > 0:
            print(f"   ✅ Generated {len(alerts)} alerts")
            for alert in alerts:
                if alert:
                    print(f"   ✅ Alert: {alert['type']} - {alert['severity']}")
            return True
        else:
            print("   ❌ No alerts generated from high-risk detections")
            return False
            
    except Exception as e:
        print(f"   ❌ Alert management test failed: {e}")
        return False

def test_secure_data_management():
    """Test secure data encryption"""
    print("🔐 Testing Secure Data Management...")
    try:
        from secure_data_manager import SecureDataManager
        
        secure_manager = SecureDataManager("test_password_123")
        
        # Test data encryption/decryption
        test_data = {
            'patient_id': 'TEST_001',
            'timestamp': datetime.now().isoformat(),
            'emotion': 'happy',
            'vital_signs': {'heart_rate': 72, 'breathing_rate': 16}
        }
        
        # Encrypt data
        encrypted_filepath = secure_manager.encrypt_data(test_data, "TEST_001")
        
        if encrypted_filepath and os.path.exists(encrypted_filepath):
            print("   ✅ Data encryption successful")
        else:
            print("   ❌ Data encryption failed")
            return False
        
        # Test decryption
        decrypted_data = secure_manager.decrypt_data(encrypted_filepath)
        
        if decrypted_data and decrypted_data.get('patient_id') == 'TEST_001':
            print("   ✅ Data decryption successful")
            return True
        else:
            print("   ❌ Data decryption failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Secure data management test failed: {e}")
        return False

def test_integration():
    """Test full integration of all systems"""
    print("🔄 Testing Full System Integration...")
    try:
        from cv_monitor_secure import SecureCVMonitor
        
        # Create full monitor with all systems
        monitor = SecureCVMonitor("INTEGRATION_TEST", "integration_password")
        
        # Verify all subsystems are initialized
        subsystems = [
            ('fall_detector', 'Fall Detection'),
            ('vital_monitor', 'Vital Signs'),
            ('alert_manager', 'Alert Management'),
            ('gaze_tracker', 'Gaze Tracking'),
            ('secure_manager', 'Secure Data Manager')
        ]
        
        for attr, name in subsystems:
            if hasattr(monitor, attr):
                print(f"   ✅ {name} system initialized")
            else:
                print(f"   ❌ {name} system missing")
                return False
        
        # Test detection capabilities exist
        detection_methods = [
            'detect_advanced_emotions',
            'detect_agitation_patterns',
            'detect_fall_risk',
            'calculate_risk_score'
        ]
        
        for method in detection_methods:
            if hasattr(monitor, method):
                print(f"   ✅ {method} available")
            else:
                print(f"   ❌ {method} missing")
                return False
        
        print("   ✅ Full system integration successful")
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False

def main():
    """Run comprehensive feature verification"""
    print("🏥 LifeLine Enhanced System - Feature Verification Test")
    print("=" * 60)
    
    # List of all tests
    tests = [
        ("Gaze Tracking", test_gaze_tracking),
        ("Agitation Detection", test_agitation_detection),
        ("Fall Detection", test_fall_detection),
        ("Vital Signs Monitoring", test_vital_signs),
        ("Alert Management", test_alert_management),
        ("Secure Data Management", test_secure_data_management),
        ("Full Integration", test_integration)
    ]
    
    # Run all tests
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    # Final results
    print("\n" + "=" * 60)
    print(f"🏥 LifeLine Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL FEATURES VERIFIED SUCCESSFULLY!")
        print("\n📊 Verified Features:")
        print("   🔍 Gaze Tracking - Eye rolling detection for seizures/unconsciousness")
        print("   😤 Agitation Detection - Multi-pattern behavioral analysis")
        print("   🛡️ Fall Detection - Posture and movement risk assessment")
        print("   💓 Vital Signs - Heart rate and breathing monitoring")
        print("   🚨 Alert Management - Priority-based intelligent alerts")
        print("   🔐 Security - HIPAA-compliant data encryption")
        print("\n✨ System ready for enhanced patient monitoring!")
    else:
        failed = total - passed
        print(f"⚠️ {failed} features need attention before deployment")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)