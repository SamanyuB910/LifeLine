# cv_monitor_secure.py - Enhanced CV Monitor with Security & Advanced Features
import os
import time
import csv
import json
from datetime import datetime, timedelta
import math
import logging

import cv2
import numpy as np
import pandas as pd

# Import our enhanced modules
from secure_data_manager import SecureDataManager
from fall_detection import FallDetectionSystem
from vital_signs import VitalSignsMonitor
from alert_manager import EnhancedAlertManager
from gaze_tracking import GazeTrackingSystem

class SecureCVMonitor:
    """
    Enhanced CV Monitor with:
    - HIPAA-compliant data encryption
    - Fall detection and prevention
    - Vital signs monitoring
    - Advanced alert management
    - Multi-patient support
    """
    
    def __init__(self, patient_id=None, master_password=None):
        # Security setup
        self.patient_id = patient_id or "default_patient"
        self.secure_manager = SecureDataManager(master_password)
        
        # Enhanced monitoring systems
        self.fall_detector = FallDetectionSystem()
        self.vital_monitor = VitalSignsMonitor()
        self.alert_manager = EnhancedAlertManager(self.patient_id)
        self.gaze_tracker = GazeTrackingSystem()
        
        # Configuration
        self.CAM_INDEX = 1
        self.SAVE_INTERVAL_SEC = 5
        self.MISSING_FACE_SEC = 3
        self.MOVEMENT_THRESHOLD_PX = 60
        
        # Initialize detectors
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Patient monitoring state
        self.patient_state = {
            'baseline_established': False,
            'last_seen': datetime.now(),
            'current_emotion': 'neutral',
            'emotion_history': [],
            'movement_patterns': [],
            'vital_signs': {},
            'risk_score': 0.0
        }
        
        logging.info(f"Secure CV Monitor initialized for patient: {self.patient_id}")
    
    def detect_advanced_emotions(self, frame, bbox):
        """Enhanced emotion detection with pain scale and micro-expressions"""
        x, y, w, h = bbox
        face_region = frame[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return "neutral", 0.5, 0  # emotion, confidence, pain_scale
        
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        height, width = gray_face.shape
        
        # Enhanced emotion detection with pain indicators
        mouth_region = gray_face[int(height*0.6):int(height*0.9), int(width*0.2):int(width*0.8)]
        eye_region = gray_face[int(height*0.2):int(height*0.5), int(width*0.1):int(width*0.9)]
        
        if mouth_region.size == 0 or eye_region.size == 0:
            return "neutral", 0.6, 0
        
        # Calculate metrics
        avg_brightness = np.mean(gray_face)
        mouth_brightness = np.mean(mouth_region)
        eye_brightness = np.mean(eye_region)
        
        # Detect edges for micro-expressions
        mouth_edges = cv2.Canny(mouth_region, 30, 100)
        eye_edges = cv2.Canny(eye_region, 20, 80)
        
        mouth_edge_density = np.sum(mouth_edges > 0) / mouth_edges.size
        eye_edge_density = np.sum(eye_edges > 0) / eye_edges.size
        
        # Pain scale detection (0-10)
        pain_indicators = 0
        if mouth_edge_density > 0.12:  # Tense mouth
            pain_indicators += 2
        if eye_edge_density > 0.08:  # Squinted eyes
            pain_indicators += 1
        if avg_brightness < 60:  # Pale face
            pain_indicators += 1
        
        pain_scale = min(10, pain_indicators)
        
        # Enhanced emotion classification
        mouth_smile_indicator = mouth_brightness > avg_brightness * 1.05
        eye_squint_indicator = eye_edge_density > 0.05
        
        if mouth_smile_indicator and eye_squint_indicator and pain_scale < 2:
            return "happy", 0.85, pain_scale
        elif pain_scale >= 6:
            return "pain", 0.80, pain_scale
        elif mouth_edge_density > 0.08 and avg_brightness > 90:
            return "surprised", 0.70, pain_scale
        elif avg_brightness < 70:
            return "sad", 0.65, pain_scale
        else:
            return "neutral", 0.80, pain_scale
    
    def estimate_head_pose(self, frame, bbox):
        """Estimate head pose from face bounding box"""
        x, y, w, h = bbox
        
        # Simple head pose estimation based on face position and size
        # This is a simplified version - real implementation would use facial landmarks
        
        # Estimate keypoints based on face rectangle
        right_eye = (x + int(w * 0.3), y + int(h * 0.35))
        left_eye = (x + int(w * 0.7), y + int(h * 0.35))
        nose = (x + int(w * 0.5), y + int(h * 0.55))
        mouth = (x + int(w * 0.5), y + int(h * 0.75))
        
        # Calculate pose angles
        rx, ry = right_eye
        lx, ly = left_eye
        nx, ny = nose

        dx = rx - lx
        dy = ry - ly
        eye_dist = math.hypot(dx, dy) + 1e-6

        # roll: tilt of the eye line
        roll = math.degrees(math.atan2(dy, dx))

        # yaw: how far nose is from eye midpoint horizontally
        eye_mid_x = (rx + lx) / 2.0
        yaw = math.degrees(math.atan2((nx - eye_mid_x), eye_dist))

        # pitch: vertical nose offset relative to eye midpoint
        eye_mid_y = (ry + ly) / 2.0
        pitch = math.degrees(math.atan2((eye_mid_y - ny), eye_dist))

        return float(yaw), float(pitch), float(roll)
    
    def detect_fall_risk(self, frame, bbox, headpose):
        """Advanced fall risk detection"""
        if bbox is None:
            return False, ["no_face"]
        
        x, y, w, h = bbox
        yaw, pitch, roll = headpose
        
        # Detect dangerous postures
        fall_risk_factors = []
        
        # Head tilted too far (potential loss of balance)
        if abs(pitch) > 45 or abs(roll) > 30:
            fall_risk_factors.append("head_tilt")
        
        # Rapid movement (potential fall)
        if hasattr(self, 'prev_bbox') and self.prev_bbox:
            prev_x, prev_y, prev_w, prev_h = self.prev_bbox
            movement = math.hypot(x - prev_x, y - prev_y)
            if movement > 100:  # Large movement
                fall_risk_factors.append("rapid_movement")
                self.patient_state['movement_patterns'].append(movement)
        
        # Face position indicates leaning/falling
        if y < 50 or y > 400:  # Face too high/low in frame
            fall_risk_factors.append("abnormal_position")
        
        self.prev_bbox = bbox
        
        return len(fall_risk_factors) > 0, fall_risk_factors
    
    def calculate_risk_score(self):
        """Calculate comprehensive patient risk score"""
        risk_factors = 0.0
        
        # Time since last seen
        time_since_seen = (datetime.now() - self.patient_state['last_seen']).total_seconds()
        if time_since_seen > 30:  # 30 seconds
            risk_factors += 0.3
        
        # Emotion-based risk
        if self.patient_state['current_emotion'] in ['pain', 'sad', 'angry']:
            risk_factors += 0.2
        
        # Movement pattern analysis
        if len(self.patient_state['movement_patterns']) > 5:
            recent_movements = self.patient_state['movement_patterns'][-5:]
            if any(m > 80 for m in recent_movements):  # Large movements
                risk_factors += 0.3
        
        # Vital signs anomalies
        if self.patient_state['vital_signs']:
            hr = self.patient_state['vital_signs'].get('heart_rate', 70)
            if hr < 50 or hr > 100:
                risk_factors += 0.2
        
        return min(1.0, risk_factors)
    
    def detect_agitation_patterns(self):
        """Detect agitation patterns based on emotion history and behavior"""
        current_time = datetime.now()
        
        # Initialize agitation result
        agitation_result = {
            'agitated': False,
            'agitation_level': 'none',
            'triggers': [],
            'duration_seconds': 0,
            'confidence': 0.0
        }
        
        # Check if we have enough emotion history
        if len(self.patient_state['emotion_history']) < 5:
            return agitation_result
        
        # Analyze recent emotions (last 2 minutes)
        recent_emotions = []
        cutoff_time = current_time - timedelta(minutes=2)
        
        for emotion_data in self.patient_state['emotion_history']:
            if emotion_data['timestamp'] >= cutoff_time:
                recent_emotions.append(emotion_data)
        
        if len(recent_emotions) < 3:
            return agitation_result
        
        # Pattern 1: Sustained negative emotions (agitation indicators)
        agitation_emotions = ['angry', 'pain', 'sad', 'surprised']
        negative_count = sum(1 for e in recent_emotions if e['emotion'] in agitation_emotions)
        negative_ratio = negative_count / len(recent_emotions)
        
        # Pattern 2: Rapid emotion changes (emotional instability)
        emotion_changes = 0
        for i in range(1, len(recent_emotions)):
            if recent_emotions[i]['emotion'] != recent_emotions[i-1]['emotion']:
                emotion_changes += 1
        change_rate = emotion_changes / max(1, len(recent_emotions) - 1)
        
        # Pattern 3: High pain scores consistently
        high_pain_count = sum(1 for e in recent_emotions if e.get('pain_scale', 0) >= 6)
        pain_ratio = high_pain_count / len(recent_emotions)
        
        # Pattern 4: Movement agitation (from movement patterns)
        movement_agitation = False
        if len(self.patient_state['movement_patterns']) >= 3:
            recent_movements = self.patient_state['movement_patterns'][-5:]
            avg_movement = np.mean(recent_movements) if recent_movements else 0
            if avg_movement > 60:  # Excessive movement
                movement_agitation = True
        
        # Calculate agitation score
        agitation_score = 0.0
        triggers = []
        
        # Sustained negative emotions
        if negative_ratio >= 0.7:  # 70% negative emotions
            agitation_score += 0.4
            triggers.append('sustained_negative_emotions')
        
        # Rapid emotion changes
        if change_rate >= 0.5:  # 50% emotion changes
            agitation_score += 0.3
            triggers.append('emotional_instability')
        
        # High pain levels
        if pain_ratio >= 0.5:  # 50% high pain
            agitation_score += 0.3
            triggers.append('persistent_pain')
        
        # Excessive movement
        if movement_agitation:
            agitation_score += 0.2
            triggers.append('excessive_movement')
        
        # Specific dangerous patterns
        # Consecutive angry emotions
        angry_streak = 0
        for emotion_data in recent_emotions[-5:]:  # Last 5 emotions
            if emotion_data['emotion'] == 'angry':
                angry_streak += 1
            else:
                break
        
        if angry_streak >= 3:
            agitation_score += 0.4
            triggers.append('sustained_anger')
        
        # Pain escalation pattern
        pain_scores = [e.get('pain_scale', 0) for e in recent_emotions[-3:]]
        if len(pain_scores) >= 3 and all(pain_scores[i] <= pain_scores[i+1] for i in range(len(pain_scores)-1)):
            if pain_scores[-1] >= 7:  # Escalating to high pain
                agitation_score += 0.3
                triggers.append('pain_escalation')
        
        # Determine agitation level
        agitation_detected = agitation_score >= 0.3
        
        if agitation_score >= 0.8:
            agitation_level = 'severe'
        elif agitation_score >= 0.6:
            agitation_level = 'high'
        elif agitation_score >= 0.3:
            agitation_level = 'moderate'
        else:
            agitation_level = 'none'
        
        # Calculate duration of agitation
        duration_seconds = 0
        if agitation_detected and recent_emotions:
            first_agitation_time = recent_emotions[0]['timestamp']
            duration_seconds = (current_time - first_agitation_time).total_seconds()
        
        # Update result
        agitation_result.update({
            'agitated': agitation_detected,
            'agitation_level': agitation_level,
            'triggers': triggers,
            'duration_seconds': duration_seconds,
            'confidence': min(1.0, agitation_score),
            'negative_emotion_ratio': negative_ratio,
            'emotion_change_rate': change_rate,
            'pain_ratio': pain_ratio,
            'movement_agitation': movement_agitation
        })
        
        return agitation_result
    
    def run_secure_monitoring(self):
        """Main secure monitoring loop"""
        cap = cv2.VideoCapture(self.CAM_INDEX)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Cannot open any camera")
        
        print(f"🔐 Secure CV Monitor started for patient: {self.patient_id}")
        print("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = datetime.now()
                
                # Face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    bbox = (x, y, w, h)
                    
                    # Update patient state
                    self.patient_state['last_seen'] = current_time
                    
                    # Advanced emotion detection with pain scale
                    emotion, confidence, pain_scale = self.detect_advanced_emotions(frame, bbox)
                    self.patient_state['current_emotion'] = emotion
                    self.patient_state['emotion_history'].append({
                        'emotion': emotion,
                        'confidence': confidence,
                        'pain_scale': pain_scale,
                        'timestamp': current_time
                    })
                    
                    # Head pose estimation
                    yaw, pitch, roll = self.estimate_head_pose(frame, bbox)
                    
                    # Fall risk detection
                    fall_risk, risk_factors = self.detect_fall_risk(frame, bbox, (yaw, pitch, roll))
                    
                    # Vital signs monitoring
                    face_region = frame[y:y+h, x:x+w]
                    vital_signs = self.vital_monitor.get_comprehensive_vitals(face_region)
                    self.patient_state['vital_signs'] = vital_signs
                    
                    # Gaze tracking for eye rolling detection
                    gaze_result = self.gaze_tracker.process_gaze_frame(face_region, bbox)
                    self.patient_state['gaze_analysis'] = gaze_result
                    
                    # Check for agitation patterns
                    agitation_result = self.detect_agitation_patterns()
                    self.patient_state['agitation_detected'] = agitation_result
                    
                    # Calculate risk score
                    risk_score = self.calculate_risk_score()
                    self.patient_state['risk_score'] = risk_score
                    
                    # Enhanced alert management
                    self.alert_manager.process_detections({
                        'emotion': emotion,
                        'pain_scale': pain_scale,
                        'fall_risk': fall_risk,
                        'risk_factors': risk_factors,
                        'vital_signs': vital_signs,
                        'risk_score': risk_score,
                        'gaze_analysis': gaze_result,
                        'agitation': agitation_result,
                        'timestamp': current_time
                    })
                    
                    # Process specific gaze alerts
                    gaze_alert = self.gaze_tracker.get_alert_message(gaze_result)
                    if gaze_alert:
                        self.alert_manager.send_alert(self.alert_manager.create_alert(gaze_alert))
                    
                    # Process agitation alerts
                    if agitation_result['agitated']:
                        agitation_alert = {
                            'type': 'patient_agitation',
                            'severity': 'high' if agitation_result['agitation_level'] in ['high', 'severe'] else 'medium',
                            'message': f"Patient agitation detected: {agitation_result['agitation_level']} level ({', '.join(agitation_result['triggers'])})",
                            'data': agitation_result
                        }
                        self.alert_manager.send_alert(self.alert_manager.create_alert(agitation_alert))
                    
                    # Secure data storage
                    self.store_secure_data({
                        'patient_id': self.patient_id,
                        'timestamp': current_time.isoformat(),
                        'emotion': emotion,
                        'confidence': confidence,
                        'pain_scale': pain_scale,
                        'yaw': yaw,
                        'pitch': pitch,
                        'roll': roll,
                        'face_x': x,
                        'face_y': y,
                        'face_w': w,
                        'face_h': h,
                        'fall_risk': fall_risk,
                        'risk_factors': risk_factors,
                        'vital_signs': vital_signs,
                        'risk_score': risk_score
                    })
                    
                    # Visual overlay with enhanced information
                    self.draw_enhanced_overlay(frame, bbox, emotion, confidence, pain_scale, 
                                             (yaw, pitch, roll), fall_risk, risk_score, vital_signs)
                
                else:
                    # No face detected - handle appropriately
                    self.handle_no_face_detection(frame, current_time)
                
                cv2.imshow(f"Secure CV Monitor - Patient: {self.patient_id}", frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                    
        except KeyboardInterrupt:
            print("Monitoring stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Final secure data backup
            self.secure_manager.log_access(f"MONITORING_SESSION_ENDED - Patient: {self.patient_id}")
    
    def store_secure_data(self, data):
        """Store data securely with encryption"""
        try:
            # Anonymize data
            anonymized_data, patient_hash = self.secure_manager.anonymize_patient_data(
                pd.DataFrame([data]), self.patient_id
            )
            
            # Encrypt and store
            self.secure_manager.encrypt_data(anonymized_data, patient_hash)
            
        except Exception as e:
            logging.error(f"Secure data storage failed: {e}")
    
    def draw_enhanced_overlay(self, frame, bbox, emotion, confidence, pain_scale, 
                            headpose, fall_risk, risk_score, vital_signs):
        """Draw enhanced overlay with all monitoring information"""
        x, y, w, h = bbox
        
        # Face bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Risk score indicator (color-coded)
        risk_color = (0, 255, 0) if risk_score < 0.3 else (0, 255, 255) if risk_score < 0.7 else (0, 0, 255)
        cv2.rectangle(frame, (10, 10), (350, 150), risk_color, -1)
        cv2.putText(frame, f"Risk Score: {risk_score:.2f}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Emotion and pain scale
        emotion_text = f"Emotion: {emotion} ({confidence:.2f})"
        cv2.putText(frame, emotion_text, (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if pain_scale > 0:
            pain_text = f"Pain Scale: {pain_scale}/10"
            cv2.putText(frame, pain_text, (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Vital signs
        hr = vital_signs.get('heart_rate', 70)
        br = vital_signs.get('breathing_rate', 16)
        cv2.putText(frame, f"HR: {hr:.0f} BPM | BR: {br:.0f} bpm", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Fall risk warning
        if fall_risk:
            cv2.putText(frame, "FALL RISK DETECTED!", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Head pose
        yaw, pitch, roll = headpose
        pose_text = f"Yaw:{yaw:.1f} Pitch:{pitch:.1f} Roll:{roll:.1f}"
        cv2.putText(frame, pose_text, (20, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Patient ID (anonymized)
        cv2.putText(frame, f"Patient: {self.patient_id[:8]}***", (frame.shape[1]-150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def handle_no_face_detection(self, frame, current_time):
        """Handle no face detection with enhanced alerting"""
        time_since_face = (current_time - self.patient_state['last_seen']).total_seconds()
        
        if time_since_face > self.MISSING_FACE_SEC:
            # High priority alert for missing patient
            self.alert_manager.create_alert({
                'type': 'patient_missing',
                'severity': 'critical',
                'message': f'Patient not visible for {time_since_face:.1f} seconds',
                'data': {'missing_time': time_since_face},
                'timestamp': current_time
            })
            
            # Visual warning
            cv2.putText(frame, "PATIENT NOT VISIBLE", (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(frame, f"Missing for {time_since_face:.1f}s", 
                       (frame.shape[1]//2 - 80, frame.shape[0]//2 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

if __name__ == "__main__":
    # Initialize with patient ID and master password
    patient_id = input("Enter Patient ID: ") or "default_patient"
    master_password = input("Enter Master Password (or press Enter for default): ") or "lifeline_secure_2024"
    
    monitor = SecureCVMonitor(patient_id, master_password)
    monitor.run_secure_monitoring()
