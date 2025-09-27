"""
LifeLine Pro - Medical Utilities
Advanced medical analysis and calculations
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

def calculate_facial_pain_features(landmarks) -> Dict[str, float]:
    """
    Calculate facial features related to pain expression using FACS
    (Facial Action Coding System)
    """
    if not landmarks:
        return {}
    
    try:
        # Extract key facial measurements
        features = {
            'brow_furrow': _calculate_brow_furrow(landmarks),
            'eye_squeeze': _calculate_eye_squeeze(landmarks),
            'nasolabial_furrow': _calculate_nasolabial_furrow(landmarks),
            'mouth_strain': _calculate_mouth_strain(landmarks)
        }
        
        return features
    except Exception as e:
        logger.error(f"Error calculating facial pain features: {e}")
        return {}

def calculate_glasgow_coma_score(eye_movement: Dict, verbal_response: Dict, motor_response: Dict) -> int:
    """
    Calculate Glasgow Coma Scale score (3-15)
    - Eye movement (1-4): No opening to Spontaneous
    - Verbal response (1-5): No response to Oriented
    - Motor response (1-6): No movement to Obeys commands
    """
    try:
        # Eye opening score
        if eye_movement.get('spontaneous', False):
            eye_score = 4
        elif eye_movement.get('to_sound', False):
            eye_score = 3
        elif eye_movement.get('to_pressure', False):
            eye_score = 2
        else:
            eye_score = 1
        
        # Verbal response score (simplified for automatic assessment)
        if verbal_response.get('oriented', False):
            verbal_score = 5
        elif verbal_response.get('confused', False):
            verbal_score = 4
        elif verbal_response.get('words', False):
            verbal_score = 3
        elif verbal_response.get('sounds', False):
            verbal_score = 2
        else:
            verbal_score = 1
        
        # Motor response score
        if motor_response.get('obeys_commands', False):
            motor_score = 6
        elif motor_response.get('localizes_pain', False):
            motor_score = 5
        elif motor_response.get('withdrawal', False):
            motor_score = 4
        elif motor_response.get('flexion', False):
            motor_score = 3
        elif motor_response.get('extension', False):
            motor_score = 2
        else:
            motor_score = 1
        
        total_score = eye_score + verbal_score + motor_score
        return total_score
    
    except Exception as e:
        logger.error(f"Error calculating Glasgow Coma Score: {e}")
        return 3  # Return minimum score on error

def calculate_rass_score(movement_data: Dict, facial_analysis: Dict) -> int:
    """
    Calculate Richmond Agitation-Sedation Scale score (-5 to +4)
    - Positive: Agitated
    - Zero: Alert and calm
    - Negative: Sedated
    """
    try:
        # Get movement intensity and patterns
        movement_intensity = movement_data.get('intensity', 0)
        movement_type = movement_data.get('movement_type', 'normal')
        
        # Get arousal indicators
        eye_open = facial_analysis.get('eyes_open', False)
        responsive = facial_analysis.get('responsive', False)
        
        # Calculate RASS score
        if movement_type == 'agitated' and movement_intensity > 0.8:
            return 4  # Combative
        elif movement_type == 'agitated' and movement_intensity > 0.6:
            return 3  # Very agitated
        elif movement_type == 'restless' and movement_intensity > 0.4:
            return 2  # Agitated
        elif movement_type == 'restless':
            return 1  # Restless
        elif movement_type == 'normal' and eye_open and responsive:
            return 0  # Alert and calm
        elif movement_type == 'minimal' and responsive:
            return -1  # Drowsy
        elif movement_type == 'minimal':
            return -2  # Light sedation
        elif movement_type == 'still' and responsive:
            return -3  # Moderate sedation
        elif movement_type == 'still':
            return -4  # Deep sedation
        else:
            return -5  # Unarousable
            
    except Exception as e:
        logger.error(f"Error calculating RASS score: {e}")
        return 0  # Return neutral score on error

def analyze_movement_patterns(current_bbox: Tuple[int, int, int, int], 
                            movement_buffer: List[Dict],
                            timestamp: float) -> Dict:
    """
    Analyze movement patterns for medical significance
    Returns movement type, intensity, and detected patterns
    """
    try:
        if len(movement_buffer) < 2:
            return {
                'movement_type': 'baseline',
                'intensity': 0,
                'patterns': {},
                'timestamp': timestamp
            }
        
        # Calculate recent movements
        recent_movements = movement_buffer[-10:]
        movements = [m.get('magnitude', 0) for m in recent_movements]
        avg_movement = np.mean(movements)
        movement_var = np.var(movements)
        
        # Classify movement
        if avg_movement < 2:
            movement_type = 'still'
            intensity = 0.1
        elif avg_movement < 5:
            movement_type = 'minimal'
            intensity = 0.3
        elif avg_movement < 15:
            movement_type = 'normal'
            intensity = 0.5
        elif avg_movement < 30:
            movement_type = 'restless'
            intensity = 0.7
        else:
            movement_type = 'agitated'
            intensity = 0.9
        
        # Detect patterns
        patterns = {
            'repetitive': movement_var < 5 and avg_movement > 10,
            'sudden': any(m > 50 for m in movements),
            'rhythmic': _is_movement_rhythmic(movements),
            'tremor': movement_var < 3 and avg_movement > 5,
            'seizure_like': movement_var > 50 and avg_movement > 20
        }
        
        return {
            'movement_type': movement_type,
            'intensity': intensity,
            'patterns': patterns,
            'average': avg_movement,
            'variance': movement_var,
            'timestamp': timestamp
        }
        
    except Exception as e:
        logger.error(f"Error analyzing movement patterns: {e}")
        return {
            'movement_type': 'error',
            'intensity': 0,
            'patterns': {},
            'timestamp': timestamp
        }

def _calculate_brow_furrow(landmarks) -> float:
    """Calculate brow furrow intensity using facial landmarks"""
    try:
        # Use MediaPipe landmark indices for eyebrows
        left_brow = [70, 63, 105, 66, 107]
        right_brow = [336, 296, 334, 293, 300]
        
        # Get average vertical positions
        left_pos = np.mean([landmarks[i].y for i in left_brow if i < len(landmarks)])
        right_pos = np.mean([landmarks[i].y for i in right_brow if i < len(landmarks)])
        
        # Calculate furrow intensity (0-1)
        intensity = min(1.0, abs(left_pos - right_pos) * 10)
        return intensity
        
    except Exception:
        return 0.0

def _calculate_eye_squeeze(landmarks) -> float:
    """Calculate eye squeeze intensity"""
    try:
        # MediaPipe eye landmark indices
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]
        
        # Calculate eye aspect ratios
        left_eye_ratio = _calculate_eye_ratio([landmarks[i] for i in left_eye if i < len(landmarks)])
        right_eye_ratio = _calculate_eye_ratio([landmarks[i] for i in right_eye if i < len(landmarks)])
        
        # Average the ratios and convert to squeeze intensity
        avg_ratio = (left_eye_ratio + right_eye_ratio) / 2
        squeeze = max(0, min(1, 1 - avg_ratio))
        
        return squeeze
        
    except Exception:
        return 0.0

def _calculate_eye_ratio(eye_points) -> float:
    """Calculate eye aspect ratio from points"""
    try:
        # Vertical distances
        v1 = abs(eye_points[1].y - eye_points[4].y)
        v2 = abs(eye_points[2].y - eye_points[5].y)
        
        # Horizontal distance
        h = abs(eye_points[0].x - eye_points[3].x)
        
        if h == 0:
            return 0
            
        return ((v1 + v2) / 2) / h
    except Exception:
        return 0.0

def _calculate_nasolabial_furrow(landmarks) -> float:
    """Calculate nasolabial furrow depth"""
    try:
        # Relevant MediaPipe landmarks
        nose_tip = landmarks[1]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        
        # Calculate average depth
        left_depth = np.sqrt((nose_tip.x - left_mouth.x)**2 + 
                           (nose_tip.y - left_mouth.y)**2)
        right_depth = np.sqrt((nose_tip.x - right_mouth.x)**2 + 
                            (nose_tip.y - right_mouth.y)**2)
        
        avg_depth = (left_depth + right_depth) / 2
        return min(1.0, avg_depth * 2)
        
    except Exception:
        return 0.0

def _calculate_mouth_strain(landmarks) -> float:
    """Calculate mouth strain/tension"""
    try:
        # Mouth corner landmarks
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        upper_lip = landmarks[0]
        lower_lip = landmarks[17]
        
        # Calculate mouth shape
        width = abs(right_corner.x - left_corner.x)
        height = abs(upper_lip.y - lower_lip.y)
        
        if width == 0:
            return 0
            
        # Calculate strain based on aspect ratio
        strain = min(1.0, (width / height) / 4)  # Normalize
        return strain
        
    except Exception:
        return 0.0

def _is_movement_rhythmic(movements: List[float], threshold: float = 0.3) -> bool:
    """
    Detect rhythmic patterns in movement data
    Using autocorrelation to find periodic signals
    """
    try:
        if len(movements) < 4:
            return False
            
        # Calculate autocorrelation
        movements = np.array(movements)
        mean = np.mean(movements)
        var = np.var(movements)
        
        if var == 0:
            return False
            
        normed_movements = (movements - mean) / var
        acorr = np.correlate(normed_movements, normed_movements, mode='full')
        acorr = acorr[len(acorr)//2:]
        
        # Check for peaks in autocorrelation
        peaks = []
        for i in range(1, len(acorr)-1):
            if acorr[i] > acorr[i-1] and acorr[i] > acorr[i+1]:
                peaks.append(acorr[i])
        
        # If we have multiple peaks above threshold, movement is rhythmic
        return len([p for p in peaks if p > threshold]) >= 2
        
    except Exception:
        return False