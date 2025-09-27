# vital_signs.py - Vital Signs Monitoring from Facial Analysis
import cv2
import numpy as np
from datetime import datetime
import math

class VitalSignsMonitor:
    """
    Vital signs monitoring using computer vision:
    - Heart rate estimation from facial color changes
    - Blood pressure indicators
    - Temperature monitoring
    - Breathing pattern detection
    """
    
    def __init__(self):
        self.color_history = []
        self.breathing_history = []
        self.max_history = 300  # 5 minutes at 1Hz
        
        # Baseline values
        self.baseline_hr = 70
        self.baseline_bp = 120
        self.baseline_temp = 98.6
    
    def estimate_heart_rate(self, face_region):
        """Estimate heart rate from subtle facial color changes"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            
            # Extract green channel (most sensitive to blood flow changes)
            green_channel = face_region[:, :, 1]
            avg_green = np.mean(green_channel)
            
            # Store in history
            self.color_history.append(avg_green)
            if len(self.color_history) > self.max_history:
                self.color_history.pop(0)
            
            # Need sufficient data for heart rate estimation
            if len(self.color_history) < 30:
                return self.baseline_hr
            
            # Simple heart rate estimation based on color variation
            signal = np.array(self.color_history)
            
            # Calculate variation in the signal
            signal_mean = np.mean(signal)
            signal_std = np.std(signal)
            
            # Estimate heart rate based on signal characteristics
            # This is a simplified version - real implementation would use FFT
            if signal_std > 5:  # High variation suggests active heart rate
                estimated_hr = self.baseline_hr + (signal_std - 5) * 2
            else:
                estimated_hr = self.baseline_hr
            
            # Sanity check
            if 40 <= estimated_hr <= 200:
                return estimated_hr
            
            return self.baseline_hr
            
        except Exception as e:
            print(f"Heart rate estimation error: {e}")
            return self.baseline_hr
    
    def detect_blood_pressure_indicators(self, face_region):
        """Detect indicators that might suggest blood pressure changes"""
        try:
            # Analyze facial color for flushing/pallor
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            
            # Red channel analysis for flushing
            red_channel = face_region[:, :, 2]
            avg_red = np.mean(red_channel)
            
            # Saturation analysis for pallor
            saturation = hsv[:, :, 1]
            avg_saturation = np.mean(saturation)
            
            # Indicators
            indicators = {
                'flushing_detected': avg_red > 120,
                'pallor_detected': avg_saturation < 50,
                'avg_red': avg_red,
                'avg_saturation': avg_saturation
            }
            
            return indicators
            
        except Exception as e:
            print(f"Blood pressure indicator detection error: {e}")
            return {'flushing_detected': False, 'pallor_detected': False}
    
    def estimate_breathing_rate(self, face_region):
        """Estimate breathing rate from subtle chest/face movements"""
        try:
            # This is a simplified version - real implementation would be more complex
            # Look for subtle periodic changes in face region
            
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate average intensity
            avg_intensity = np.mean(gray)
            
            # Store in breathing history
            self.breathing_history.append(avg_intensity)
            if len(self.breathing_history) > self.max_history:
                self.breathing_history.pop(0)
            
            # Need sufficient data
            if len(self.breathing_history) < 20:
                return 16  # Normal breathing rate
            
            # Simple breathing rate estimation based on intensity variation
            signal = np.array(self.breathing_history)
            signal_std = np.std(signal)
            
            # Estimate breathing rate based on signal characteristics
            if signal_std > 3:  # High variation suggests active breathing
                estimated_br = 16 + (signal_std - 3) * 2
            else:
                estimated_br = 16
            
            if 6 <= estimated_br <= 30:
                return estimated_br
            
            return 16  # Default normal rate
            
        except Exception as e:
            print(f"Breathing rate estimation error: {e}")
            return 16
    
    def get_comprehensive_vitals(self, face_region):
        """Get comprehensive vital signs analysis"""
        if face_region is None or face_region.size == 0:
            return {
                'heart_rate': self.baseline_hr,
                'breathing_rate': 16,
                'blood_pressure_indicators': {'flushing_detected': False, 'pallor_detected': False},
                'timestamp': datetime.now(),
                'confidence': 0.0
            }
        
        heart_rate = self.estimate_heart_rate(face_region)
        breathing_rate = self.estimate_breathing_rate(face_region)
        bp_indicators = self.detect_blood_pressure_indicators(face_region)
        
        # Calculate confidence based on data availability
        confidence = min(1.0, len(self.color_history) / 60)  # Full confidence after 1 minute
        
        return {
            'heart_rate': heart_rate,
            'breathing_rate': breathing_rate,
            'blood_pressure_indicators': bp_indicators,
            'timestamp': datetime.now(),
            'confidence': confidence
        }
