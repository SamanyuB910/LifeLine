"""
LifeLine Pro - Enhanced Data Models
Core data structures for medical monitoring
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
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

@dataclass
class PatientProfile:
    """Comprehensive patient information"""
    patient_id: str
    name: str
    age: int
    gender: str
    room: str
    admission_date: datetime
    primary_diagnosis: str
    comorbidities: List[str]
    medications: List[str]
    allergies: List[str]
    emergency_contact: str
    attending_physician: str
    nurse_assigned: str
    
    def get_risk_factors(self) -> List[str]:
        """Identify risk factors based on patient profile"""
        risks = []
        if self.age > 65:
            risks.append("Advanced age")
        if "Diabetes" in self.comorbidities:
            risks.append("Diabetic complications")
        if "Cardiac" in self.primary_diagnosis or any("cardiac" in c.lower() for c in self.comorbidities):
            risks.append("Cardiac risk")
        return risks

@dataclass
class AnalysisResult:
    """Comprehensive analysis result from monitoring"""
    timestamp: datetime
    face_detected: bool
    emotion_state: Optional[EmotionState]
    pain_score: float
    vitals: Optional[VitalSigns]
    movement_data: Optional[Dict]
    insights: List[Dict]
    alerts: List[Dict]
    risk_score: int = 0
    risk_factors: Dict[str, float] = None

    def is_critical(self) -> bool:
        """Check if the current state requires immediate attention"""
        return (
            self.risk_score > 80 or
            (self.vitals and self.vitals.is_critical()) or
            self.pain_score > 8 or
            (self.emotion_state in [EmotionState.SEVERE_DISTRESS, EmotionState.AGITATED])
        )