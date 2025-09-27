# alert_manager.py - Enhanced Alert Management System
import json
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

class EnhancedAlertManager:
    """
    Advanced alert management system with:
    - Priority-based alert routing
    - Escalation protocols
    - Alert fatigue prevention
    - Custom thresholds per patient
    - Integration capabilities
    """
    
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.alerts_dir = "secure_alerts"
        os.makedirs(self.alerts_dir, exist_ok=True)
        
        # Alert configuration
        self.alert_thresholds = {
            'pain_scale': 7,
            'fall_risk': 0.7,
            'missing_time': 30,
            'heart_rate_high': 100,
            'heart_rate_low': 50,
            'breathing_rate_high': 25,
            'breathing_rate_low': 8
        }
        
        # Alert history for fatigue prevention
        self.recent_alerts = []
        self.alert_cooldowns = {}  # Prevent spam alerts
        
        # Escalation settings
        self.escalation_times = {
            'critical': 60,  # 1 minute
            'high': 300,     # 5 minutes
            'medium': 900,   # 15 minutes
            'low': 1800      # 30 minutes
        }
        
        logging.basicConfig(
            filename=os.path.join(self.alerts_dir, f"alerts_{patient_id}.log"),
            level=logging.INFO
        )
    
    def process_detections(self, detections: Dict):
        """Process all detections and create appropriate alerts"""
        current_time = datetime.now()
        new_alerts = []
        
        # Pain scale alert
        pain_scale = detections.get('pain_scale', 0)
        if pain_scale >= self.alert_thresholds['pain_scale']:
            new_alerts.append(self.create_alert({
                'type': 'high_pain',
                'severity': 'high',
                'message': f'Patient experiencing high pain level: {pain_scale}/10',
                'data': {'pain_scale': pain_scale},
                'timestamp': current_time
            }))
        
        # Fall risk alert
        fall_risk = detections.get('fall_risk', False)
        if fall_risk:
            risk_factors = detections.get('risk_factors', [])
            new_alerts.append(self.create_alert({
                'type': 'fall_risk',
                'severity': 'critical',
                'message': f'Fall risk detected: {", ".join(risk_factors)}',
                'data': {'risk_factors': risk_factors},
                'timestamp': current_time
            }))
        
        # Vital signs alerts
        vital_signs = detections.get('vital_signs', {})
        if vital_signs:
            # Heart rate alerts
            hr = vital_signs.get('heart_rate', 70)
            if hr > self.alert_thresholds['heart_rate_high']:
                new_alerts.append(self.create_alert({
                    'type': 'high_heart_rate',
                    'severity': 'high',
                    'message': f'Elevated heart rate: {hr:.0f} BPM',
                    'data': {'heart_rate': hr},
                    'timestamp': current_time
                }))
            elif hr < self.alert_thresholds['heart_rate_low']:
                new_alerts.append(self.create_alert({
                    'type': 'low_heart_rate',
                    'severity': 'high',
                    'message': f'Low heart rate: {hr:.0f} BPM',
                    'data': {'heart_rate': hr},
                    'timestamp': current_time
                }))
            
            # Breathing rate alerts
            br = vital_signs.get('breathing_rate', 16)
            if br > self.alert_thresholds['breathing_rate_high']:
                new_alerts.append(self.create_alert({
                    'type': 'rapid_breathing',
                    'severity': 'medium',
                    'message': f'Rapid breathing: {br:.0f} breaths/min',
                    'data': {'breathing_rate': br},
                    'timestamp': current_time
                }))
            elif br < self.alert_thresholds['breathing_rate_low']:
                new_alerts.append(self.create_alert({
                    'type': 'slow_breathing',
                    'severity': 'high',
                    'message': f'Slow breathing: {br:.0f} breaths/min',
                    'data': {'breathing_rate': br},
                    'timestamp': current_time
                }))
        
        # Gaze tracking alerts
        gaze_analysis = detections.get('gaze_analysis', {})
        if gaze_analysis and gaze_analysis.get('eye_roll_detected'):
            severity = 'critical' if gaze_analysis.get('sustained_abnormal_gaze') else 'high'
            new_alerts.append(self.create_alert({
                'type': 'eye_rolling_detected',
                'severity': severity,
                'message': f'Eye rolling detected (confidence: {gaze_analysis.get("rolling_confidence", 0):.2f})',
                'data': gaze_analysis,
                'timestamp': current_time
            }))
        
        # Agitation alerts
        agitation = detections.get('agitation', {})
        if agitation and agitation.get('agitated'):
            agitation_level = agitation.get('agitation_level', 'moderate')
            severity = 'critical' if agitation_level == 'severe' else 'high'
            triggers = agitation.get('triggers', [])
            new_alerts.append(self.create_alert({
                'type': 'patient_agitation',
                'severity': severity,
                'message': f'Patient agitation detected: {agitation_level} level ({", ".join(triggers)})',
                'data': agitation,
                'timestamp': current_time
            }))
        
        # Overall risk score alert
        risk_score = detections.get('risk_score', 0.0)
        if risk_score >= 0.8:
            new_alerts.append(self.create_alert({
                'type': 'high_risk',
                'severity': 'critical',
                'message': f'High overall risk score: {risk_score:.2f}',
                'data': {'risk_score': risk_score},
                'timestamp': current_time
            }))
        
        # Process and send alerts
        for alert in new_alerts:
            if alert:
                self.send_alert(alert)
        
        return new_alerts
    
    def create_alert(self, alert_data: Dict) -> Optional[Dict]:
        """Create a new alert with proper validation and cooldown"""
        alert_type = alert_data['type']
        current_time = datetime.now()
        
        # Check cooldown period to prevent alert fatigue
        if alert_type in self.alert_cooldowns:
            last_alert_time = self.alert_cooldowns[alert_type]
            cooldown_period = 300  # 5 minutes default cooldown
            if (current_time - last_alert_time).total_seconds() < cooldown_period:
                return None  # Skip alert due to cooldown
        
        # Create alert
        alert = {
            'id': f"{self.patient_id}_{alert_type}_{current_time.strftime('%Y%m%d_%H%M%S')}",
            'patient_id': self.patient_id,
            'type': alert_type,
            'severity': alert_data['severity'],
            'message': alert_data['message'],
            'data': alert_data.get('data', {}),
            'timestamp': current_time.isoformat(),
            'status': 'active',
            'acknowledged': False,
            'escalated': False
        }
        
        # Update cooldown
        self.alert_cooldowns[alert_type] = current_time
        
        # Store alert
        self.store_alert(alert)
        
        return alert
    
    def send_alert(self, alert: Dict):
        """Send alert through appropriate channels"""
        try:
            # Log alert
            logging.info(f"ALERT_SENT - {alert['type']} - {alert['severity']} - {alert['message']}")
            
            # Store in recent alerts
            self.recent_alerts.append(alert)
            if len(self.recent_alerts) > 100:
                self.recent_alerts.pop(0)
            
            # Here you would integrate with actual alert systems:
            # - Hospital communication systems
            # - Mobile notifications
            # - Email alerts
            # - Voice announcements
            # - Integration with nurse call systems
            
            print(f"🚨 ALERT: {alert['severity'].upper()} - {alert['message']}")
            
            # Schedule escalation if not acknowledged
            self.schedule_escalation(alert)
            
        except Exception as e:
            logging.error(f"Alert sending failed: {e}")
    
    def store_alert(self, alert: Dict):
        """Store alert in secure format"""
        try:
            alert_file = os.path.join(self.alerts_dir, f"alert_{alert['id']}.json")
            with open(alert_file, 'w') as f:
                json.dump(alert, f, indent=2)
        except Exception as e:
            logging.error(f"Alert storage failed: {e}")
    
    def schedule_escalation(self, alert: Dict):
        """Schedule alert escalation if not acknowledged"""
        # In a real implementation, this would set up timers
        # For now, we'll just log the escalation schedule
        escalation_time = self.escalation_times.get(alert['severity'], 1800)
        escalation_time_str = (datetime.now() + timedelta(seconds=escalation_time)).isoformat()
        
        logging.info(f"ESCALATION_SCHEDULED - Alert: {alert['id']} - Escalation time: {escalation_time_str}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Mark alert as acknowledged"""
        try:
            alert_file = os.path.join(self.alerts_dir, f"alert_{alert_id}.json")
            if os.path.exists(alert_file):
                with open(alert_file, 'r') as f:
                    alert = json.load(f)
                
                alert['acknowledged'] = True
                alert['acknowledged_by'] = acknowledged_by
                alert['acknowledged_at'] = datetime.now().isoformat()
                
                with open(alert_file, 'w') as f:
                    json.dump(alert, f, indent=2)
                
                logging.info(f"ALERT_ACKNOWLEDGED - {alert_id} by {acknowledged_by}")
                return True
        except Exception as e:
            logging.error(f"Alert acknowledgment failed: {e}")
            return False
    
    def get_active_alerts(self):
        """Get all active alerts for the patient"""
        active_alerts = []
        try:
            for filename in os.listdir(self.alerts_dir):
                if filename.startswith(f"alert_{self.patient_id}_") and filename.endswith('.json'):
                    alert_file = os.path.join(self.alerts_dir, filename)
                    with open(alert_file, 'r') as f:
                        alert = json.load(f)
                    
                    if alert.get('status') == 'active':
                        active_alerts.append(alert)
        except Exception as e:
            logging.error(f"Failed to get active alerts: {e}")
        
        return active_alerts
    
    def get_alert_summary(self):
        """Get summary of alerts for dashboard"""
        active_alerts = self.get_active_alerts()
        
        summary = {
            'total_active': len(active_alerts),
            'by_severity': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'by_type': {},
            'recent_alerts': active_alerts[-5:]  # Last 5 alerts
        }
        
        for alert in active_alerts:
            severity = alert.get('severity', 'low')
            alert_type = alert.get('type', 'unknown')
            
            summary['by_severity'][severity] += 1
            summary['by_type'][alert_type] = summary['by_type'].get(alert_type, 0) + 1
        
        return summary
