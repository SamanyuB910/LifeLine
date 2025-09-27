"""
LifeLine Pro - Enhanced Data Models
Core data structures for medical monitoring
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

class AlertSeverity(Enum):
    """Alert severity levels following medical standards"""
    CRITICAL = "critical"  # Immediate intervention required
    HIGH = "high"         # Urgent attention needed
    MEDIUM = "medium"     # Monitor closely
    LOW = "low"          # Informational
    RESOLVED = "resolved" # Issue resolved

class EmotionState(Enum):
    """Patient emotional states with medical context"""
    COMFORTABLE = "comfortable"
    NEUTRAL = "neutral"
    MILD_DISTRESS = "mild_distress"
    MODERATE_DISTRESS = "moderate_distress"
    SEVERE_DISTRESS = "severe_distress"
    AGITATED = "agitated"
    UNRESPONSIVE = "unresponsive"

class TrendDirection(Enum):
    """Trend directions for analytics"""
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class VitalSigns:
    """Medical vital signs data structure"""
    heart_rate: int
    spo2: int
    respiratory_rate: int
    blood_pressure_systolic: int
    blood_pressure_diastolic: int
    temperature: float
    timestamp: datetime
    
    def is_critical(self) -> bool:
        """Check if any vital sign is in critical range"""
        return (
            self.heart_rate < 40 or self.heart_rate > 140 or
            self.spo2 < 90 or
            self.respiratory_rate < 8 or self.respiratory_rate > 30 or
            self.blood_pressure_systolic < 90 or self.blood_pressure_systolic > 180 or
            self.temperature < 95 or self.temperature > 104
        )
    
    def get_abnormal_vitals(self) -> List[str]:
        """Return list of abnormal vital signs"""
        abnormal = []
        if self.heart_rate < 60 or self.heart_rate > 100:
            abnormal.append("heart_rate")
        if self.spo2 < 95:
            abnormal.append("spo2")
        if self.respiratory_rate < 12 or self.respiratory_rate > 20:
            abnormal.append("respiratory_rate")
        if (self.blood_pressure_systolic < 100 or 
            self.blood_pressure_systolic > 140):
            abnormal.append("blood_pressure")
        if self.temperature < 97 or self.temperature > 99:
            abnormal.append("temperature")
        return abnormal

@dataclass
class PatientProfile:
    """Enhanced patient profile with medical history"""
    patient_id: str
    name: str
    room: str
    age: int
    medical_history: List[str]
    monitoring_protocol: str
    admission_date: datetime = field(default_factory=datetime.now)
    discharge_date: Optional[datetime] = None
    allergies: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    primary_diagnosis: Optional[str] = None
    secondary_diagnoses: List[str] = field(default_factory=list)
    care_notes: List[Dict] = field(default_factory=list)

@dataclass
class AnalysisResult:
    """Comprehensive analysis result with analytics data"""
    timestamp: datetime
    face_detected: bool
    emotion_state: Optional[EmotionState]
    pain_score: float
    vitals: Optional[VitalSigns]
    movement_data: Optional[Dict]
    insights: List[Dict]
    alerts: List[Dict]
    risk_score: int
    risk_factors: Dict = field(default_factory=dict)
    analytics: Dict[str, Any] = field(default_factory=dict)  # Analytics results
    trends: Dict[str, TrendDirection] = field(default_factory=dict)
    predictions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalyticsTrend:
    """Analytics trend data"""
    metric: str
    direction: TrendDirection
    magnitude: float
    timeframe: str
    start_value: float
    end_value: float
    confidence: float
    insights: List[Dict] = field(default_factory=list)

@dataclass
class Prediction:
    """Predictive analytics result"""
    metric: str
    predicted_value: float
    confidence: float
    timeframe: str
    contributing_factors: List[str]
    risk_level: str
    timestamp: datetime = field(default_factory=datetime.now)
        """Get list of abnormal vital signs"""
        abnormal = []
        if self.heart_rate < 60 or self.heart_rate > 100:
            abnormal.append(f"HR: {self.heart_rate}")
        if self.spo2 < 95:
            abnormal.append(f"SpO2: {self.spo2}%")
        if self.respiratory_rate < 12 or self.respiratory_rate > 20:
            abnormal.append(f"RR: {self.respiratory_rate}")
        if self.blood_pressure_systolic < 100 or self.blood_pressure_systolic > 140:
            abnormal.append(f"BP: {self.blood_pressure_systolic}/{self.blood_pressure_diastolic}")
        if self.temperature < 97 or self.temperature > 99.5:
            abnormal.append(f"Temp: {self.temperature}°F")
        return abnormal



