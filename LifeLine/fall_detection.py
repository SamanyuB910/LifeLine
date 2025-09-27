# fall_detection.py - Advanced Fall Detection and Prevention System
import cv2
import numpy as np
from datetime import datetime, timedelta
import math
import os

# Optional ML imports - fallback gracefully if not available
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    from PIL import Image
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("ML dependencies not available. Using rule-based detection only.")

class MLFallDetector:
    """Machine Learning-based fall detection enhancement"""
    
    def __init__(self, model_path='fall_model.json'):
        self.ml_available = True  # Always available now (simple model)
        self.model = None
        self.confidence_threshold = 0.6
        
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Load trained ML model"""
        try:
            import json
            
            if os.path.exists(model_path):
                with open(model_path, 'r') as f:
                    model_data = json.load(f)
                self.model = model_data['model']
                print(f"✅ Simple ML model loaded from {model_path}")
            else:
                print(f"⚠️ ML model not found at {model_path}, using default thresholds")
                # Default simple model
                self.model = {
                    'aspect_ratio': {'threshold': 1.5, 'higher_indicates_fall': False},
                    'bottom_heavy': {'threshold': 1.0, 'higher_indicates_fall': True}
                }
            
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            print(f"❌ Failed to load ML model: {e}")
            self.ml_available = False
    
    def predict_fall_risk(self, frame):
        """Predict fall risk using simple ML model"""
        if not self.ml_available or self.model is None:
            return {'fall_detected': False, 'confidence': 0.0, 'method': 'unavailable'}
        
        try:
            # Extract simple features from frame
            features = self._extract_features(frame)
            
            # Simple model prediction
            prediction = self._predict_fall(features)
            
            return {
                'fall_detected': prediction['is_fall'],
                'confidence': prediction['confidence'],
                'details': prediction['details'],
                'method': 'simple_ml'
            }
        
        except Exception as e:
            print(f"ML prediction error: {e}")
            return {'fall_detected': False, 'confidence': 0.0, 'method': 'error'}
    
    def _extract_features(self, frame):
        """Extract simple features for fall detection"""
        h, w, _ = frame.shape
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Basic features
        features = {
            'aspect_ratio': w / h,
            'brightness': np.mean(gray),
            'contrast': np.std(gray),
            'bottom_heavy': np.mean(gray[h//2:, :]) / np.mean(gray[:h//2, :]) if np.mean(gray[:h//2, :]) > 0 else 1.0,
            'horizontal_edges': len(cv2.findContours(cv2.Canny(gray, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        }
        
        return features
    
    def _predict_fall(self, features):
        """Predict if features indicate a fall"""
        fall_score = 0
        total_features = len(self.model)
        
        details = {}
        
        for feature_name, feature_model in self.model.items():
            if feature_name not in features:
                continue
                
            feature_value = features[feature_name]
            threshold = feature_model['threshold']
            higher_indicates_fall = feature_model['higher_indicates_fall']
            
            if higher_indicates_fall:
                contributes_to_fall = feature_value > threshold
            else:
                contributes_to_fall = feature_value < threshold
            
            if contributes_to_fall:
                fall_score += 1
            
            details[feature_name] = {
                'value': feature_value,
                'threshold': threshold,
                'contributes_to_fall': contributes_to_fall
            }
        
        confidence = fall_score / total_features if total_features > 0 else 0.0
        is_fall = confidence > 0.5  # Majority vote
        
        return {
            'is_fall': is_fall,
            'confidence': confidence,
            'fall_score': fall_score,
            'total_features': total_features,
            'details': details
        }

class FallDetectionSystem:
    """
    Advanced fall detection system with:
    - Posture analysis
    - Movement pattern recognition
    - Fall prediction algorithms
    - Bed exit detection
    - ML-enhanced detection
    """
    
    def __init__(self, enable_ml=True):
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
        
        # ML enhancement
        self.ml_detector = MLFallDetector() if enable_ml else None
    
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
    
    def enhanced_fall_prediction(self, frame, bbox=None, headpose=None):
        """
        Enhanced fall prediction combining rule-based and ML approaches
        
        Args:
            frame: OpenCV frame
            bbox: Face bounding box (x, y, w, h) or None
            headpose: Head pose angles (yaw, pitch, roll) or None
            
        Returns:
            dict: Comprehensive fall risk assessment
        """
        result = {
            'fall_detected': False,
            'risk_score': 0.0,
            'rule_based_score': 0.0,
            'ml_score': 0.0,
            'combined_score': 0.0,
            'confidence': 0.0,
            'method': 'hybrid',
            'details': {}
        }
        
        # Rule-based analysis
        if bbox is not None and headpose is not None:
            posture_analysis = self.analyze_posture(bbox, headpose)
            rule_based_score = posture_analysis['risk_score']
            
            # Extract tilt features
            tilt_method, tilt_angle = self.extract_tilt_feature(frame, bbox)
            if abs(tilt_angle) > 20:  # Significant tilt
                rule_based_score += 0.2
            
            result['rule_based_score'] = rule_based_score
            result['details']['posture'] = posture_analysis
            result['details']['tilt_angle'] = tilt_angle
            result['details']['tilt_method'] = tilt_method
        
        # ML-based analysis
        if self.ml_detector is not None:
            ml_result = self.ml_detector.predict_fall_risk(frame)
            ml_contribution = 0.3 if ml_result['fall_detected'] else 0.0
            
            result['ml_score'] = ml_contribution
            result['details']['ml_result'] = ml_result
        
        # Combine scores
        combined_score = min(1.0, result['rule_based_score'] + result['ml_score'])
        result['combined_score'] = combined_score
        result['risk_score'] = combined_score
        
        # Final decision
        result['fall_detected'] = combined_score >= self.fall_risk_threshold
        result['confidence'] = combined_score
        
        # Store in history
        self.posture_history.append({
            'timestamp': datetime.now(),
            'risk_score': combined_score,
            'fall_detected': result['fall_detected']
        })
        
        # Keep history manageable
        if len(self.posture_history) > 100:
            self.posture_history = self.posture_history[-50:]
        
        return result
    
    @staticmethod
    def extract_tilt_feature(frame, bbox=None):
        '''Returns (method, tilt_angle), method is 'face' or 'pose', angle is degrees from vertical.'''
        # Try face-based
        if bbox is not None:
            x, y, w, h = bbox
            cx = x + w // 2
            cy = y + h // 2
            # Head tilt angle not directly available; estimate from face aspect ratio for frontal faces
            angle = 0  # Default to vertical
            if w > 0 and h > 0:
                aspect = w / h
                # If face is wider than tall, likely tilting
                if aspect > 1.35:
                    angle = 25
                elif aspect < 0.75:
                    angle = -25
            return ('face', angle)
        # If bbox is None, try OpenCV DNN Pose estimation for torso axis
        try:
            protoFile = cv2.data.haarcascades + "pose_deploy_linevec.prototxt"
            weightsFile = cv2.data.haarcascades + "pose_iter_440000.caffemodel"
            if not (os.path.exists(protoFile) and os.path.exists(weightsFile)):
                return ('none', 0)
            net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
            H, W = frame.shape[:2]
            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
            net.setInput(inpBlob)
            output = net.forward()
            points = []
            for i in range(15):  # Use up to 15 keypoints
                probMap = output[0, i, :, :]
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                x = (W * point[0]) / output.shape[3]
                y = (H * point[1]) / output.shape[2]
                if prob > 0.1:
                    points.append((int(x), int(y)))
                else:
                    points.append(None)
            # Neck (1), R shoulder (2), L shoulder (5), R hip (8), L hip (11)
            shoulders = [points[2], points[5]]
            hips = [points[8], points[11]]
            valid_sh = [p for p in shoulders if p]
            valid_hip = [p for p in hips if p]
            if len(valid_sh) == 2 and len(valid_hip) == 2:
                mean_shoulder = ((valid_sh[0][0] + valid_sh[1][0])//2, (valid_sh[0][1] + valid_sh[1][1])//2)
                mean_hip = ((valid_hip[0][0] + valid_hip[1][0])//2, (valid_hip[0][1] + valid_hip[1][1])//2)
                dx = mean_hip[0] - mean_shoulder[0]
                dy = mean_hip[1] - mean_shoulder[1]
                tilt_angle = math.degrees(math.atan2(dx, dy))
                return ('pose', tilt_angle)
        except Exception:
            pass
        return ('none', 0)
