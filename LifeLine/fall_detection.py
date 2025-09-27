# fall_detection.py - Advanced Fall Detection and Prevention System
import cv2
import numpy as np
from datetime import datetime, timedelta
import math

class FallDetectionSystem:
    """
    Advanced fall detection system with:
    - Posture analysis
    - Movement pattern recognition
    - Fall prediction algorithms
    - Bed exit detection
    """
    
    def __init__(self):
        self.movement_history = []
        self.posture_history = []
        self.fall_risk_threshold = 0.7
        self.movement_window = 10  # seconds
        
        # Fall risk indicators
        self.risk_factors = {
            'rapid_movement': 0.3,
            'abnormal_posture': 0.4,
            'bed_exit_attempt': 0.6,
            'loss_of_balance': 0.8
        }
    
    def analyze_posture(self, bbox, headpose):
        """Analyze patient posture for fall risk"""
        if bbox is None:
            return {'posture': 'unknown', 'risk_score': 0.0}
        
        x, y, w, h = bbox
        yaw, pitch, roll = headpose
        
        risk_score = 0.0
        posture = 'normal'
        
        # Head tilt analysis
        if abs(pitch) > 45:
            risk_score += 0.3
            posture = 'head_tilted'
        
        if abs(roll) > 30:
            risk_score += 0.2
            posture = 'head_rolled'
        
        # Face position analysis
        face_center_y = y + h // 2
        
        if face_center_y < 100:  # Face too high (potential standing/falling)
            risk_score += 0.4
            posture = 'standing_risk'
        elif face_center_y > 300:  # Face too low (potential lying down)
            risk_score += 0.2
            posture = 'lying_down'
        
        # Face size analysis (distance from camera)
        face_area = w * h
        if face_area < 5000:  # Face too small (far from camera)
            risk_score += 0.3
            posture = 'too_far'
        elif face_area > 20000:  # Face too large (too close)
            risk_score += 0.2
            posture = 'too_close'
        
        return {
            'posture': posture,
            'risk_score': min(1.0, risk_score),
            'head_tilt': abs(pitch),
            'face_position': face_center_y,
            'face_size': face_area
        }
    
    def detect_bed_exit(self, bbox, movement_history):
        """Detect potential bed exit attempts"""
        if bbox is None or len(movement_history) < 3:
            return False, 0.0
        
        # Analyze recent movement patterns
        recent_movements = movement_history[-5:] if len(movement_history) >= 5 else movement_history
        
        # Look for movement toward edge of frame
        x, y, w, h = bbox
        frame_width = 640  # Assuming standard camera resolution
        
        # Movement toward left edge (common bed exit direction)
        edge_distance = min(x, frame_width - x - w)
        if edge_distance < 50:  # Close to edge
            return True, 0.6
        
        # Rapid movement pattern
        if len(recent_movements) >= 3:
            movement_velocity = sum(recent_movements) / len(recent_movements)
            if movement_velocity > 80:  # High velocity
                return True, 0.4
        
        return False, 0.0
    
    def predict_fall_risk(self, current_analysis):
        """Predict fall risk based on multiple factors"""
        risk_score = 0.0
        
        # Posture risk
        posture_risk = current_analysis.get('posture_risk', 0.0)
        risk_score += posture_risk * 0.4
        
        # Movement risk
        movement_risk = current_analysis.get('movement_risk', 0.0)
        risk_score += movement_risk * 0.3
        
        # Bed exit risk
        bed_exit_risk = current_analysis.get('bed_exit_risk', 0.0)
        risk_score += bed_exit_risk * 0.3
        
        return min(1.0, risk_score)
    
    def should_alert(self, risk_score):
        """Determine if fall alert should be triggered"""
        return risk_score >= self.fall_risk_threshold
