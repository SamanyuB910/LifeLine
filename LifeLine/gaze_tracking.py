# gaze_tracking.py - Advanced Gaze Tracking for Eye Rolling Detection
import cv2
import numpy as np
import math
from datetime import datetime, timedelta
import logging

class GazeTrackingSystem:
    """
    Advanced gaze tracking system for detecting:
    - Eye rolling (upward, backward movement)
    - Abnormal eye movements
    - Seizure-related eye symptoms
    - Loss of consciousness indicators
    - Pupil dilation/constriction
    """
    
    def __init__(self):
        # Eye detection cascades
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
        self.right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
        
        # Tracking history
        self.eye_position_history = []
        self.pupil_history = []
        self.gaze_pattern_history = []
        self.max_history = 60  # Track last 60 frames (~2 seconds at 30fps)
        
        # Thresholds for abnormal gaze patterns
        self.thresholds = {
            'eye_roll_angle': 15,  # degrees of upward movement to consider rolling
            'rapid_movement_threshold': 20,  # pixels/frame for rapid eye movement
            'sustained_abnormal_duration': 3.0,  # seconds for sustained abnormal gaze
            'pupil_dilation_ratio': 1.5,  # ratio change for abnormal pupil size
            'eye_closure_duration': 2.0  # seconds for concerning eye closure
        }
        
        # Current state
        self.current_state = {
            'eye_roll_detected': False,
            'abnormal_gaze': False,
            'eye_closure_prolonged': False,
            'pupil_abnormal': False,
            'last_normal_gaze': datetime.now()
        }
        
        logging.info("Gaze Tracking System initialized")
    
    def detect_eyes(self, face_region):
        """Detect and locate eyes in face region"""
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes using multiple cascades for better accuracy
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 5, minSize=(20, 20))
        left_eyes = self.left_eye_cascade.detectMultiScale(gray_face, 1.1, 5, minSize=(20, 20))
        right_eyes = self.right_eye_cascade.detectMultiScale(gray_face, 1.1, 5, minSize=(20, 20))
        
        # Combine detections and filter
        all_eyes = []
        
        # Add general eye detections
        for (x, y, w, h) in eyes:
            all_eyes.append({'x': x, 'y': y, 'w': w, 'h': h, 'type': 'general'})
        
        # Add specific left/right eye detections
        for (x, y, w, h) in left_eyes:
            all_eyes.append({'x': x, 'y': y, 'w': w, 'h': h, 'type': 'left'})
        
        for (x, y, w, h) in right_eyes:
            all_eyes.append({'x': x, 'y': y, 'w': w, 'h': h, 'type': 'right'})
        
        # Sort by size and position to get best candidates
        if len(all_eyes) >= 2:
            # Sort by x position to get left and right eye
            all_eyes.sort(key=lambda e: e['x'])
            return all_eyes[:2]  # Return top 2 eyes
        
        return all_eyes
    
    def analyze_eye_position(self, eye_region, eye_bbox):
        """Analyze eye position and detect rolling patterns"""
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_eye = clahe.apply(gray_eye)
        
        height, width = enhanced_eye.shape
        
        # Detect pupil/iris using circular Hough transform
        circles = cv2.HoughCircles(
            enhanced_eye,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(min(width, height) * 0.3),
            param1=50,
            param2=30,
            minRadius=int(min(width, height) * 0.1),
            maxRadius=int(min(width, height) * 0.4)
        )
        
        pupil_info = None
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            if len(circles) > 0:
                # Get the most centered circle
                center_x, center_y = width // 2, height // 2
                best_circle = min(circles, key=lambda c: math.sqrt((c[0] - center_x)**2 + (c[1] - center_y)**2))
                
                pupil_x, pupil_y, pupil_radius = best_circle
                pupil_info = {
                    'center': (pupil_x, pupil_y),
                    'radius': pupil_radius,
                    'relative_position': (pupil_x / width, pupil_y / height)
                }
        
        # If no pupil detected, use intensity-based detection
        if pupil_info is None:
            # Find darkest region (likely pupil/iris)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(enhanced_eye)
            pupil_info = {
                'center': min_loc,
                'radius': 10,  # estimated
                'relative_position': (min_loc[0] / width, min_loc[1] / height)
            }
        
        return pupil_info
    
    def detect_eye_rolling(self, eye_positions):
        """Detect eye rolling patterns from eye positions"""
        if not eye_positions or len(self.eye_position_history) < 10:
            return False, 0.0
        
        current_positions = []
        for eye_pos in eye_positions:
            if eye_pos:
                rel_x, rel_y = eye_pos['relative_position']
                current_positions.append((rel_x, rel_y))
        
        if not current_positions:
            return False, 0.0
        
        # Add to history
        self.eye_position_history.append({
            'timestamp': datetime.now(),
            'positions': current_positions
        })
        
        # Keep only recent history
        if len(self.eye_position_history) > self.max_history:
            self.eye_position_history.pop(0)
        
        # Analyze recent movement patterns
        recent_positions = self.eye_position_history[-10:]  # Last 10 frames
        
        # Calculate upward movement trend
        upward_movement = 0
        for i in range(1, len(recent_positions)):
            prev_positions = recent_positions[i-1]['positions']
            curr_positions = recent_positions[i]['positions']
            
            for j, (prev_pos, curr_pos) in enumerate(zip(prev_positions, curr_positions)):
                if prev_pos and curr_pos:
                    prev_x, prev_y = prev_pos
                    curr_x, curr_y = curr_pos
                    
                    # Check for upward movement (rolling back)
                    if curr_y < prev_y:  # Y decreases = upward movement
                        upward_movement += (prev_y - curr_y)
        
        # Calculate rolling confidence
        avg_upward_per_frame = upward_movement / max(1, len(recent_positions) - 1)
        rolling_confidence = min(1.0, avg_upward_per_frame * 10)  # Scale to 0-1
        
        # Detect sustained upward gaze (eye rolling)
        eye_roll_detected = False
        if rolling_confidence > 0.3:  # Threshold for eye rolling
            # Check if eyes are positioned high in the socket
            latest_positions = recent_positions[-1]['positions']
            high_position_count = 0
            
            for pos in latest_positions:
                if pos[1] < 0.3:  # Y position in upper 30% of eye region
                    high_position_count += 1
            
            if high_position_count >= len(latest_positions) * 0.5:  # At least 50% of eyes
                eye_roll_detected = True
        
        return eye_roll_detected, rolling_confidence
    
    def analyze_pupil_changes(self, eye_positions):
        """Analyze pupil size changes for medical indicators"""
        if not eye_positions:
            return {'normal': True, 'dilation_ratio': 1.0}
        
        current_pupils = []
        for eye_pos in eye_positions:
            if eye_pos and 'radius' in eye_pos:
                current_pupils.append(eye_pos['radius'])
        
        if not current_pupils:
            return {'normal': True, 'dilation_ratio': 1.0}
        
        avg_pupil_size = np.mean(current_pupils)
        
        # Add to history
        self.pupil_history.append({
            'timestamp': datetime.now(),
            'size': avg_pupil_size
        })
        
        # Keep only recent history
        if len(self.pupil_history) > self.max_history:
            self.pupil_history.pop(0)
        
        # Calculate baseline pupil size
        if len(self.pupil_history) >= 20:
            recent_sizes = [p['size'] for p in self.pupil_history[-20:]]
            baseline_size = np.median(recent_sizes)
            current_ratio = avg_pupil_size / baseline_size
            
            # Check for abnormal dilation/constriction
            abnormal = current_ratio > self.thresholds['pupil_dilation_ratio'] or \
                      current_ratio < (1.0 / self.thresholds['pupil_dilation_ratio'])
            
            return {
                'normal': not abnormal,
                'dilation_ratio': current_ratio,
                'baseline_size': baseline_size,
                'current_size': avg_pupil_size
            }
        
        return {'normal': True, 'dilation_ratio': 1.0}
    
    def process_gaze_frame(self, face_region, face_bbox):
        """Process a single frame for gaze tracking"""
        if face_region is None or face_region.size == 0:
            return self._create_empty_result()
        
        # Detect eyes in face region
        detected_eyes = self.detect_eyes(face_region)
        
        if not detected_eyes:
            return self._create_empty_result()
        
        # Analyze each detected eye
        eye_analyses = []
        for eye_info in detected_eyes:
            ex, ey, ew, eh = eye_info['x'], eye_info['y'], eye_info['w'], eye_info['h']
            
            # Extract eye region
            eye_region = face_region[ey:ey+eh, ex:ex+ew]
            
            if eye_region.size > 0:
                eye_position = self.analyze_eye_position(eye_region, (ex, ey, ew, eh))
                eye_analyses.append(eye_position)
        
        # Detect eye rolling
        eye_roll_detected, rolling_confidence = self.detect_eye_rolling(eye_analyses)
        
        # Analyze pupil changes
        pupil_analysis = self.analyze_pupil_changes(eye_analyses)
        
        # Update current state
        self.current_state['eye_roll_detected'] = eye_roll_detected
        self.current_state['pupil_abnormal'] = not pupil_analysis['normal']
        
        if not eye_roll_detected and pupil_analysis['normal']:
            self.current_state['last_normal_gaze'] = datetime.now()
        
        # Check for sustained abnormal gaze
        time_since_normal = (datetime.now() - self.current_state['last_normal_gaze']).total_seconds()
        sustained_abnormal = time_since_normal > self.thresholds['sustained_abnormal_duration']
        
        return {
            'eye_roll_detected': eye_roll_detected,
            'rolling_confidence': rolling_confidence,
            'pupil_analysis': pupil_analysis,
            'sustained_abnormal_gaze': sustained_abnormal,
            'eyes_detected': len(detected_eyes),
            'time_since_normal_sec': time_since_normal,
            'gaze_alert_level': self._calculate_alert_level(eye_roll_detected, rolling_confidence, sustained_abnormal, pupil_analysis)
        }
    
    def _calculate_alert_level(self, eye_roll, confidence, sustained, pupil_analysis):
        """Calculate overall alert level for gaze abnormalities"""
        if eye_roll and sustained:
            return 'critical'  # Eye rolling for extended period
        elif eye_roll and confidence > 0.7:
            return 'high'  # Strong eye rolling detection
        elif sustained and not pupil_analysis['normal']:
            return 'high'  # Sustained abnormal gaze with pupil issues
        elif eye_roll or not pupil_analysis['normal']:
            return 'medium'  # Either eye rolling or pupil abnormality
        else:
            return 'low'  # Normal gaze
    
    def _create_empty_result(self):
        """Create empty result when no eyes detected"""
        return {
            'eye_roll_detected': False,
            'rolling_confidence': 0.0,
            'pupil_analysis': {'normal': True, 'dilation_ratio': 1.0},
            'sustained_abnormal_gaze': False,
            'eyes_detected': 0,
            'time_since_normal_sec': 0.0,
            'gaze_alert_level': 'low'
        }
    
    def get_alert_message(self, gaze_result):
        """Generate appropriate alert message based on gaze analysis"""
        if not gaze_result:
            return None
        
        alert_level = gaze_result['gaze_alert_level']
        
        if alert_level == 'critical':
            return {
                'type': 'eye_rolling_critical',
                'severity': 'critical',
                'message': 'CRITICAL: Sustained eye rolling detected - possible seizure or loss of consciousness',
                'data': gaze_result
            }
        elif alert_level == 'high':
            if gaze_result['eye_roll_detected']:
                return {
                    'type': 'eye_rolling_detected',
                    'severity': 'high',
                    'message': f'Eye rolling detected (confidence: {gaze_result["rolling_confidence"]:.2f})',
                    'data': gaze_result
                }
            else:
                return {
                    'type': 'abnormal_gaze',
                    'severity': 'high',
                    'message': 'Abnormal gaze pattern detected',
                    'data': gaze_result
                }
        elif alert_level == 'medium':
            return {
                'type': 'gaze_concern',
                'severity': 'medium',
                'message': 'Unusual eye movement or pupil activity detected',
                'data': gaze_result
            }
        
        return None