"""
LifeLine Pro - Enhanced Medical Monitor
Core monitoring system with advanced medical features
"""

import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import json
import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import queue

from .data_models import (
    AlertSeverity,
    EmotionState,
    VitalSigns,
    PatientProfile,
    AnalysisResult,
    AnalyticsResult
)
from .analytics import HealthAnalytics
from .analytics import HealthAnalytics
from ..utils.medical_utils import (
    calculate_facial_pain_features,
    calculate_glasgow_coma_score,
    calculate_rass_score,
    analyze_movement_patterns
)

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

logger = logging.getLogger(__name__)

class MedicalGradeMonitor:
    """Enhanced medical-grade patient monitoring system"""
    
    def __init__(self, patient: PatientProfile):
        self.patient = patient
        
        # Load configuration
        self._load_config()
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        
        # Initialize buffers
        buffer_sizes = self.config['analysis']
        self.emotion_buffer = deque(maxlen=buffer_sizes['emotion_buffer_size'])
        self.movement_buffer = deque(maxlen=buffer_sizes['movement_buffer_size'])
        self.vital_buffer = deque(maxlen=buffer_sizes['vital_buffer_size'])
        self.alert_queue = queue.PriorityQueue()
        self.analysis_buffer = deque(maxlen=100)  # Store recent analysis results
        
        # Initialize analytics
        self.analytics = HealthAnalytics()
        
        # State tracking
        self.pain_score = 0
        self.glasgow_coma_scale = 15  # Start at max (normal)
        self.rass_score = 0  # Start at neutral
        self.last_risk_update = datetime.now()
        
        logger.info(f"Initialized medical-grade monitoring for patient {patient.patient_id}")
    
    def _load_config(self):
        """Load monitoring configuration and thresholds"""
        try:
            config_dir = Path(__file__).parent.parent / 'config'
            
            with open(config_dir / 'settings.json', 'r') as f:
                self.config = json.load(f)
            
            with open(config_dir / 'thresholds.json', 'r') as f:
                self.thresholds = json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Use default configuration
            self.config = {
                'monitoring': {
                    'face_mesh': {
                        'max_num_faces': 1,
                        'refine_landmarks': True,
                        'min_detection_confidence': 0.5,
                        'min_tracking_confidence': 0.5
                    },
                    'face_detection': {
                        'model_selection': 1,
                        'min_detection_confidence': 0.5
                    }
                },
                'analysis': {
                    'emotion_buffer_size': 300,
                    'movement_buffer_size': 60,
                    'vital_buffer_size': 1440,
                    'analysis_fps': 10,
                    'risk_score_update_interval': 5
                }
            }
            self.thresholds = {
                'vital_thresholds': {
                    'heart_rate': {'critical_low': 40, 'low': 60, 'high': 100, 'critical_high': 140},
                    'spo2': {'critical_low': 90, 'low': 95}
                }
            }
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, AnalysisResult]:
        """
        Process a single frame with comprehensive medical analysis
        Returns: (annotated_frame, analysis_results)
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize result
        analysis = AnalysisResult(
            timestamp=datetime.now(),
            face_detected=False,
            emotion_state=None,
            pain_score=0,
            vitals=None,
            movement_data=None,
            insights=[],
            alerts=[],
            risk_score=0
        )
        
        # Detect face
        face_detection = self.mp_face_detection.process(rgb_frame)
        
        if face_detection.detections:
            detection = face_detection.detections[0]
            bbox_c = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            
            bbox = (
                int(bbox_c.xmin * iw),
                int(bbox_c.ymin * ih),
                int(bbox_c.width * iw),
                int(bbox_c.height * ih)
            )
            
            analysis.face_detected = True
            
            # Get face crop for detailed analysis
            x, y, w, h = bbox
            face_crop = frame[y:y+h, x:x+w]
            
            # Analyze facial features
            face_mesh = self.mp_face_mesh.process(rgb_frame)
            if face_mesh.multi_face_landmarks:
                landmarks = face_mesh.multi_face_landmarks[0].landmark
                
                # Pain analysis
                pain_features = calculate_facial_pain_features(landmarks)
                self.pain_score = self._calculate_pain_score(pain_features)
                analysis.pain_score = self.pain_score
                
                # Movement analysis
                movement_data = analyze_movement_patterns(
                    bbox, list(self.movement_buffer), 
                    datetime.now().timestamp()
                )
                self.movement_buffer.append(movement_data)
                analysis.movement_data = movement_data
                
                # Consciousness assessment (GCS)
                eye_movement = self._analyze_eye_movement(landmarks)
                verbal_response = {'oriented': True}  # Simplified - would need audio
                motor_response = {'obeys_commands': True}  # Simplified
                self.glasgow_coma_scale = calculate_glasgow_coma_score(
                    eye_movement, verbal_response, motor_response
                )
                
                # Agitation assessment (RASS)
                self.rass_score = calculate_rass_score(
                    movement_data,
                    {'eyes_open': eye_movement.get('spontaneous', False)}
                )
            
            # Emotion analysis
            if DEEPFACE_AVAILABLE and face_crop.size > 0:
                try:
                    emotion_state, confidence = self._analyze_emotion(face_crop)
                    self.emotion_buffer.append({
                        'state': emotion_state,
                        'confidence': confidence,
                        'timestamp': datetime.now()
                    })
                    analysis.emotion_state = emotion_state
                except Exception as e:
                    logger.debug(f"Emotion analysis error: {e}")
            
            # Calculate comprehensive risk score
            if (datetime.now() - self.last_risk_update).seconds >= self.config['analysis']['risk_score_update_interval']:
                risk_score, risk_factors = self._calculate_risk_score()
                analysis.risk_score = risk_score
                analysis.risk_factors = risk_factors
                self.last_risk_update = datetime.now()
            
            # Store analysis result
            self.analysis_buffer.append(analysis)
            
            # Run analytics
            analytics_results = self.analytics.analyze_trends(
                self.patient,
                list(self.analysis_buffer)
            )
            
            # Update analysis with analytics insights
            analysis.analytics = analytics_results
            
            # Generate medical insights (combine monitor and analytics insights)
            monitor_insights = self._generate_insights(analysis)
            analytics_insights = analytics_results.get('insights', [])
            analysis.insights = monitor_insights + analytics_insights
            
            # Generate alerts if needed
            alerts = self._check_alert_conditions(analysis)
            analysis.alerts = alerts
            
            # Save for historical analysis
            self.analytics.save_analysis_result(analysis)
            
            # Draw annotations
            frame = self._draw_medical_overlay(frame, bbox, analysis)
        
        return frame, analysis
    
    def _calculate_pain_score(self, features: Dict[str, float]) -> float:
        """Calculate pain score (0-10) from facial features"""
        if not features:
            return 0
            
        # Get thresholds
        thresholds = self.thresholds['pain_detection']
        
        # Calculate weighted score
        score = 0
        if features['brow_furrow'] > thresholds['brow_furrow_threshold']:
            score += 3
        if features['eye_squeeze'] > thresholds['eye_squeeze_threshold']:
            score += 2.5
        if features['nasolabial_furrow'] > thresholds['nasolabial_furrow_threshold']:
            score += 2.5
        if features['mouth_strain'] > thresholds['mouth_strain_threshold']:
            score += 2
            
        return min(10, score)
    
    def _analyze_eye_movement(self, landmarks) -> Dict:
        """Analyze eye movement and openness"""
        try:
            # Eye landmark indices
            left_eye = [33, 160, 158, 133, 153, 144]
            right_eye = [362, 385, 387, 263, 373, 380]
            
            # Calculate eye aspect ratios
            left_ratio = self._eye_aspect_ratio([landmarks[i] for i in left_eye])
            right_ratio = self._eye_aspect_ratio([landmarks[i] for i in right_eye])
            
            avg_ratio = (left_ratio + right_ratio) / 2
            
            return {
                'spontaneous': avg_ratio > 0.25,
                'to_sound': avg_ratio > 0.15,
                'to_pressure': avg_ratio > 0.1
            }
            
        except Exception as e:
            logger.debug(f"Eye movement analysis error: {e}")
            return {}
    
    def _eye_aspect_ratio(self, eye_points) -> float:
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
        except:
            return 0
    
    def _analyze_emotion(self, face_image: np.ndarray) -> Tuple[EmotionState, float]:
        """Analyze facial emotion and map to medical states"""
        result = DeepFace.analyze(
            face_image,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
        
        emotion_scores = result[0]['emotion']
        dominant_emotion = result[0]['dominant_emotion']
        confidence = emotion_scores[dominant_emotion] / 100.0
        
        # Map to medical states
        if dominant_emotion == 'happy':
            state = EmotionState.COMFORTABLE
        elif dominant_emotion == 'neutral':
            state = EmotionState.NEUTRAL
        elif dominant_emotion in ['sad', 'fear']:
            state = EmotionState.MILD_DISTRESS if confidence < 0.7 else EmotionState.MODERATE_DISTRESS
        elif dominant_emotion == 'angry':
            state = EmotionState.AGITATED
        elif dominant_emotion in ['disgust', 'surprise']:
            state = EmotionState.SEVERE_DISTRESS if self.pain_score > 5 else EmotionState.MILD_DISTRESS
        else:
            state = EmotionState.NEUTRAL
        
        return state, confidence
    
    def _calculate_risk_score(self) -> Tuple[int, Dict[str, float]]:
        """Calculate comprehensive medical risk score"""
        factors = {}
        
        # Pain component (0-25)
        factors['pain'] = min(25, self.pain_score * 2.5)
        
        # Consciousness component (0-25)
        gcs_risk = max(0, (15 - self.glasgow_coma_scale) * 1.67)
        factors['consciousness'] = gcs_risk
        
        # Agitation component (0-20)
        if self.rass_score >= 2:  # Agitated
            factors['agitation'] = min(20, abs(self.rass_score) * 5)
        elif self.rass_score <= -3:  # Very sedated
            factors['agitation'] = min(20, abs(self.rass_score) * 4)
        else:
            factors['agitation'] = 0
        
        # Emotional distress (0-15)
        if len(self.emotion_buffer) > 0:
            latest_emotion = self.emotion_buffer[-1]['state']
            if latest_emotion == EmotionState.SEVERE_DISTRESS:
                factors['emotional'] = 15
            elif latest_emotion == EmotionState.MODERATE_DISTRESS:
                factors['emotional'] = 10
            elif latest_emotion == EmotionState.MILD_DISTRESS:
                factors['emotional'] = 5
            else:
                factors['emotional'] = 0
        
        # Movement risk (0-15)
        if len(self.movement_buffer) > 0:
            movement = self.movement_buffer[-1]
            if movement.get('patterns', {}).get('seizure_like', False):
                factors['movement'] = 15
            elif movement.get('movement_type') == 'agitated':
                factors['movement'] = 10
            elif movement.get('movement_type') == 'restless':
                factors['movement'] = 5
            else:
                factors['movement'] = 0
        
        total_risk = sum(factors.values())
        
        # Apply time-based modifiers
        current_hour = datetime.now().hour
        if 22 <= current_hour or current_hour <= 6:
            total_risk *= 1.1  # 10% increase at night
        
        return min(100, int(total_risk)), factors
    
    def _generate_insights(self, analysis: AnalysisResult) -> List[Dict]:
        """Generate actionable medical insights"""
        insights = []
        
        # Critical conditions
        if analysis.risk_score > 80:
            insights.append({
                'priority': 'CRITICAL',
                'title': 'Immediate Attention Required',
                'message': f'Risk score {analysis.risk_score}% - Multiple concerning indicators',
                'actions': ['Notify attending physician', 'Increase monitoring frequency']
            })
        
        # Pain management
        if self.pain_score > 7:
            insights.append({
                'priority': 'HIGH',
                'title': 'Severe Pain Detected',
                'message': f'Pain score {self.pain_score}/10 - Facial indicators of severe pain',
                'actions': ['Assess pain', 'Review pain medication']
            })
        elif self.pain_score > 5:
            insights.append({
                'priority': 'MEDIUM',
                'title': 'Moderate Pain Indicated',
                'message': f'Pain score {self.pain_score}/10',
                'actions': ['Monitor pain progression', 'Consider PRN medication']
            })
        
        # Consciousness/Sedation
        if self.glasgow_coma_scale < 15:
            insights.append({
                'priority': 'HIGH',
                'title': 'Decreased Consciousness',
                'message': f'GCS Score: {self.glasgow_coma_scale}/15',
                'actions': ['Neurological assessment', 'Review medication effects']
            })
        
        if abs(self.rass_score) >= 2:
            state = 'Agitated' if self.rass_score > 0 else 'Sedated'
            insights.append({
                'priority': 'MEDIUM',
                'title': f'RASS Score: {state}',
                'message': f'RASS Score: {self.rass_score}',
                'actions': ['Assess agitation/sedation', 'Review care plan']
            })
        
        # Movement concerns
        if analysis.movement_data:
            if analysis.movement_data.get('patterns', {}).get('seizure_like'):
                insights.append({
                    'priority': 'CRITICAL',
                    'title': 'Possible Seizure Activity',
                    'message': 'Abnormal movement patterns detected',
                    'actions': ['Immediate assessment', 'Protect airways', 'Prepare emergency medication']
                })
        
        return insights[:5]  # Return top 5 insights
    
    def _check_alert_conditions(self, analysis: AnalysisResult) -> List[Dict]:
        """Check conditions that require alerts"""
        alerts = []
        current_time = datetime.now()
        
        # Risk score alert
        if analysis.risk_score > 80:
            alerts.append(self._create_alert(
                AlertSeverity.CRITICAL,
                "Critical Risk Score",
                f"Patient risk score: {analysis.risk_score}%",
                ['risk']
            ))
        
        # Pain alert
        if self.pain_score > 7:
            alerts.append(self._create_alert(
                AlertSeverity.HIGH,
                "Severe Pain Detected",
                f"Pain score: {self.pain_score}/10",
                ['pain']
            ))
        
        # Consciousness alert
        if self.glasgow_coma_scale < 13:
            alerts.append(self._create_alert(
                AlertSeverity.HIGH,
                "Decreased Consciousness",
                f"GCS Score: {self.glasgow_coma_scale}/15",
                ['consciousness']
            ))
        
        # Agitation alert
        if abs(self.rass_score) >= 3:
            state = "Agitation" if self.rass_score > 0 else "Deep Sedation"
            alerts.append(self._create_alert(
                AlertSeverity.MEDIUM,
                f"Significant {state}",
                f"RASS Score: {self.rass_score}",
                ['agitation']
            ))
        
        return alerts
    
    def _create_alert(self, severity: AlertSeverity, title: str, message: str, 
                     categories: List[str]) -> Dict:
        """Create a structured alert"""
        alert = {
            'id': int(datetime.now().timestamp() * 1000),
            'severity': severity.value,
            'title': title,
            'message': message,
            'categories': categories,
            'timestamp': datetime.now().isoformat(),
            'patient_id': self.patient.patient_id,
            'patient_name': self.patient.name,
            'room': self.patient.room
        }
        
        # Add to queue
        priority = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        self.alert_queue.put((priority[severity.value], alert))
        
        return alert
    
    def _draw_medical_overlay(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                            analysis: AnalysisResult) -> np.ndarray:
        """Draw medical information overlay on frame"""
        x, y, w, h = bbox
        
        # Draw face box with color based on risk
        if analysis.risk_score > 80:
            color = (0, 0, 255)  # Red
        elif analysis.risk_score > 60:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
            
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Add key metrics
        metrics = [
            f"Risk: {analysis.risk_score}%",
            f"Pain: {self.pain_score:.1f}/10",
            f"GCS: {self.glasgow_coma_scale}/15",
            f"RASS: {self.rass_score}"
        ]
        
        for i, metric in enumerate(metrics):
            cv2.putText(frame, metric, (x, y - 10 - (i * 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add emotion state if available
        if analysis.emotion_state:
            emotion_text = f"State: {analysis.emotion_state.value}"
            cv2.putText(frame, emotion_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame