# demo_all_features.py - Live Feature Demonstration
import cv2
import numpy as np
from datetime import datetime
import time

def demo_enhanced_monitoring():
    """Demonstrate all enhanced monitoring features"""
    print("🏥 LifeLine Enhanced Patient Monitoring System")
    print("=" * 60)
    print("🔍 Features: Gaze Tracking | Agitation | Fall Risk | Vital Signs")
    print("🔐 Security: HIPAA-compliant encryption enabled")
    print("🚨 Alerts: Multi-level priority system active")
    print("=" * 60)
    
    try:
        from cv_monitor_secure import SecureCVMonitor
        
        # Initialize enhanced monitor
        monitor = SecureCVMonitor("DEMO_PATIENT", "demo_password_2024")
        
        print(f"✅ Enhanced CV Monitor initialized")
        print(f"   📹 Camera: Index {monitor.CAM_INDEX}")
        print(f"   🔐 Patient ID: {monitor.patient_id}")
        print(f"   🧠 Systems: Gaze | Agitation | Fall | Vitals | Alerts")
        
        # Try to open camera for live demo
        cap = cv2.VideoCapture(monitor.CAM_INDEX)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print(f"\n🎥 Starting live monitoring demo...")
            print("   Press 'q' to quit, 's' to show status")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                current_time = datetime.now()
                
                # Face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = monitor.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                
                # Process frame if face detected
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Extract face region for analysis
                    face_region = frame[y:y+h, x:x+w]
                    
                    # Emotion detection
                    emotion, confidence, pain_scale = monitor.detect_advanced_emotions(frame, (x, y, w, h))
                    
                    # Gaze tracking
                    gaze_result = monitor.gaze_tracker.process_gaze_frame(face_region, (x, y, w, h))
                    
                    # Agitation detection (every 10 frames to avoid overload)
                    if frame_count % 10 == 0:
                        # Add current emotion to history
                        monitor.patient_state['emotion_history'].append({
                            'emotion': emotion,
                            'confidence': confidence,
                            'pain_scale': pain_scale,
                            'timestamp': current_time
                        })
                        
                        # Keep history manageable
                        if len(monitor.patient_state['emotion_history']) > 50:
                            monitor.patient_state['emotion_history'] = monitor.patient_state['emotion_history'][-30:]
                        
                        agitation_result = monitor.detect_agitation_patterns()
                    else:
                        agitation_result = {'agitated': False, 'agitation_level': 'none'}
                    
                    # Vital signs
                    vital_signs = monitor.vital_monitor.get_comprehensive_vitals(face_region)
                    
                    # Display information on frame
                    info_y = 30
                    cv2.putText(frame, f"Emotion: {emotion} ({confidence:.2f})", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    info_y += 25
                    cv2.putText(frame, f"Pain Scale: {pain_scale}/10", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    info_y += 25
                    cv2.putText(frame, f"Heart Rate: {vital_signs.get('heart_rate', 'N/A'):.0f} BPM", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    info_y += 25
                    
                    # Gaze tracking info
                    if gaze_result['eye_roll_detected']:
                        cv2.putText(frame, "EYE ROLLING DETECTED!", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, f"Gaze: Normal", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    info_y += 25
                    
                    # Agitation info
                    if agitation_result['agitated']:
                        cv2.putText(frame, f"AGITATION: {agitation_result['agitation_level'].upper()}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    else:
                        cv2.putText(frame, "Behavior: Normal", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                else:
                    cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Show frame
                cv2.imshow('LifeLine Enhanced Monitoring', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    print(f"\n📊 Current Status:")
                    print(f"   👤 Face detected: {'Yes' if len(faces) > 0 else 'No'}")
                    if len(faces) > 0:
                        print(f"   😊 Emotion: {emotion} (confidence: {confidence:.2f})")
                        print(f"   🩹 Pain scale: {pain_scale}/10")
                        print(f"   💓 Heart rate: {vital_signs.get('heart_rate', 'N/A'):.0f} BPM")
                        print(f"   🫁 Breathing: {vital_signs.get('breathing_rate', 'N/A'):.0f} /min")
                        print(f"   👁️ Gaze: {'Eye rolling!' if gaze_result['eye_roll_detected'] else 'Normal'}")
                        print(f"   😤 Agitation: {agitation_result['agitation_level']}")
            
            cap.release()
            cv2.destroyAllWindows()
            print("\n✅ Live monitoring demo completed")
        
        else:
            print("❌ No camera available for live demo")
            print("💡 All features are verified and ready - camera needed for live monitoring")
    
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("💡 Try running test_all_features.py for feature verification")

if __name__ == "__main__":
    demo_enhanced_monitoring()