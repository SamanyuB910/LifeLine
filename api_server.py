"""
Enhanced LifeLine API Server
============================

This module provides REST API endpoints and WebSocket integration 
to connect the existing Python CV monitoring backend with the Next.js frontend.

Features:
- REST API endpoints matching frontend expectations
- WebSocket for real-time updates
- Alert management integration with existing CSV system
- Patient monitoring data aggregation
"""

import json
import csv
import os
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'lifeline_secret_key'
CORS(app, origins=["http://localhost:3000", "http://localhost:3001"])  # Support both ports
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:3000", "http://localhost:3001"])

class LifeLineBackendAPI:
    def __init__(self):
        self.alerts_file = "alerts.csv"
        self.predictions_file = "emotion_predictions.csv"
        self.is_monitoring = False
        
    def get_alerts_by_status(self):
        """Get alerts grouped by status matching frontend structure"""
        alerts = {'ongoing': [], 'resolved': [], 'previous': []}
        
        if os.path.exists(self.alerts_file):
            try:
                with open(self.alerts_file, 'r', newline='', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        alert = {
                            'id': row.get('timestamp', str(time.time())),
                            'type': self.map_severity_to_frontend_type(row.get('severity', 'medium')),
                            'category': self.map_alert_category(row.get('alert_type', '')),
                            'title': self.format_alert_title(row.get('alert_type', 'Unknown Alert')),
                            'message': row.get('message', ''),
                            'timestamp': row.get('timestamp', ''),
                            'duration': self.format_duration(row.get('duration', '0')),
                            'severity': row.get('severity', 'medium'),
                            'acknowledged': row.get('status') in ['resolved', 'previous'],
                            'patient': 'Sarah Johnson',
                            'room': 'ICU-204'
                        }
                        
                        status = row.get('status', 'ongoing')
                        if status in alerts:
                            alerts[status].append(alert)
            except Exception as e:
                print(f"Error reading alerts: {e}")
        
        return alerts
    
    def get_latest_emotion_data(self):
        """Get the latest emotion prediction from CSV"""
        emotion_data = {
            'emotion': 'neutral',
            'confidence': 0.0,
            'timestamp': datetime.now().isoformat(),
            'face_detected': False
        }
        
        if os.path.exists(self.predictions_file):
            try:
                with open(self.predictions_file, 'r', newline='', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    rows = list(reader)
                    if rows:
                        latest = rows[-1]
                        
                        # Clean up emotion field - handle tuple format like "('happy', 0.85)"
                        emotion_raw = latest.get('emotion', 'neutral')
                        if emotion_raw.startswith("('") and emotion_raw.endswith("')"):
                            # Extract emotion from tuple format
                            emotion_clean = emotion_raw.split("'")[1]
                        else:
                            emotion_clean = emotion_raw
                        
                        emotion_data = {
                            'emotion': emotion_clean,
                            'confidence': float(latest.get('confidence', 0.0)),
                            'timestamp': latest.get('timestamp', datetime.now().isoformat()),
                            'face_detected': latest.get('face_detected', 'False') == 'True'
                        }
            except Exception as e:
                print(f"Error reading emotion data: {e}")
        
        return emotion_data
    
    def get_patient_monitoring_data(self):
        """Get comprehensive patient data for frontend dashboard"""
        emotion_data = self.get_latest_emotion_data()
        alerts = self.get_alerts_by_status()
        
        # Check if we need to generate a "no face detected" alert
        if emotion_data.get('emotion') == 'no_face' and not emotion_data.get('face_detected', False):
            # Check if there's already a recent "no face detected" alert to avoid duplicates
            existing_no_face_alerts = [
                alert for alert in alerts['ongoing'] 
                if 'No Face Detected' in alert.get('title', '') or 'face detected' in alert.get('message', '').lower()
            ]
            
            if not existing_no_face_alerts:
                # Generate a high-risk "no face detected" alert
                self.generate_no_face_alert()
                # Refresh alerts after generating new one
                alerts = self.get_alerts_by_status()
        
        # Calculate fall risk based on ongoing alerts and emotion
        ongoing_count = len(alerts['ongoing'])
        fall_risk = min(ongoing_count * 20 + 15, 95)
        
        # Determine attention level based on face detection
        attention_level = 85 if emotion_data.get('face_detected', False) else 30
        
        return {
            'id': 'patient_001',
            'name': 'Sarah Johnson',
            'room': 'ICU-204',
            'status': 'stable' if ongoing_count == 0 else 'requires_attention',
            'emotion': {
                'current': emotion_data.get('emotion', 'neutral'),
                'confidence': emotion_data.get('confidence', 0.0),
                'timestamp': emotion_data.get('timestamp', datetime.now().isoformat())
            },
            'vitals': {
                'fall_risk': fall_risk,
                'attention_level': attention_level,
                'movement_activity': 'normal' if ongoing_count < 2 else 'elevated',
                'posture': 'sitting upright' if emotion_data.get('face_detected') else 'not detected'
            },
            'alerts': alerts,
            'camera_status': 'active' if self.is_monitoring else 'inactive',
            'last_update': datetime.now().isoformat()
        }
    
    def map_severity_to_frontend_type(self, severity):
        """Map backend severity to frontend alert types"""
        mapping = {
            'critical': 'critical',
            'high': 'high',
            'medium': 'medium', 
            'low': 'low'
        }
        return mapping.get(severity.lower(), 'medium')
    
    def generate_no_face_alert(self):
        """Generate a high-risk alert when no face is detected"""
        import csv
        import os
        
        alert_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'alert_type': 'No Face Detected',
            'message': 'Patient not visible in camera view - immediate attention required',
            'severity': 'high',
            'patient': 'Sarah Johnson',
            'room': 'ICU-204',
            'status': 'ongoing',
            'acknowledged': False
        }
        
        # Save to alerts.csv
        alerts_file = os.path.join(os.path.dirname(__file__), 'alerts.csv')
        file_exists = os.path.exists(alerts_file)
        
        with open(alerts_file, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['timestamp', 'alert_type', 'message', 'severity', 'patient', 'room', 'status', 'acknowledged']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(alert_data)
        
        print(f"🚨 Generated high-risk alert: No Face Detected")
    
    def map_alert_category(self, alert_type):
        """Map alert types to frontend categories"""
        alert_lower = alert_type.lower()
        if 'face' in alert_lower or 'no face' in alert_lower:
            return 'behavior'
        elif 'emotion' in alert_lower:
            return 'behavior'
        elif 'movement' in alert_lower or 'fall' in alert_lower:
            return 'fall_risk'
        elif 'vital' in alert_lower or 'heart' in alert_lower:
            return 'vital_signs'
        elif 'system' in alert_lower:
            return 'system'
        else:
            return 'system'
    
    def format_alert_title(self, alert_type):
        """Format alert type into a readable title"""
        return alert_type.replace('_', ' ').title()
    
    def format_duration(self, duration_str):
        """Format duration for frontend display"""
        try:
            duration_seconds = float(duration_str)
            if duration_seconds < 60:
                return f"{int(duration_seconds)}s"
            elif duration_seconds < 3600:
                return f"{int(duration_seconds/60)}m"
            else:
                return f"{int(duration_seconds/3600)}h"
        except:
            return "0s"

# Initialize API
api = LifeLineBackendAPI()

# API Routes matching frontend expectations
@app.route('/api/patients/patient_001')
def get_patient_data():
    """Get specific patient data"""
    try:
        return jsonify(api.get_patient_monitoring_data())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts')
def get_all_alerts():
    """Get all alerts grouped by status"""
    try:
        return jsonify(api.get_alerts_by_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/ongoing')
def get_ongoing_alerts():
    """Get ongoing alerts"""
    try:
        alerts = api.get_alerts_by_status()
        return jsonify({'alerts': alerts['ongoing']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/resolved')
def get_resolved_alerts():
    """Get resolved alerts"""
    try:
        alerts = api.get_alerts_by_status()
        return jsonify({'alerts': alerts['resolved']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/previous')
def get_previous_alerts():
    """Get previous alerts"""
    try:
        alerts = api.get_alerts_by_status()
        return jsonify({'alerts': alerts['previous']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system-status')
def get_system_status():
    """Get system health status"""
    try:
        return jsonify({
            'status': 'operational',
            'uptime': '24h 15m',
            'camera_connected': True,
            'ai_model_active': True,
            'monitoring_active': api.is_monitoring,
            'last_update': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/video-stream-status')
def get_video_status():
    """Get video stream status"""
    try:
        return jsonify({
            'active': api.is_monitoring,
            'resolution': '1080p',
            'fps': 30,
            'quality': 'HD'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/latest-snapshot')
def get_latest_snapshot():
    """Serve the latest camera snapshot"""
    try:
        from flask import send_file
        import os
        
        snapshot_path = os.path.join(os.path.dirname(__file__), 'snapshots', 'latest.jpg')
        
        if os.path.exists(snapshot_path):
            return send_file(snapshot_path, mimetype='image/jpeg')
        else:
            # Return a placeholder if no snapshot exists
            return jsonify({'error': 'No snapshot available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket events for real-time updates
@socketio.on('connect')
def handle_connect():
    print('Frontend client connected')
    emit('status', {'message': 'Connected to LifeLine monitoring system'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Frontend client disconnected')

def broadcast_updates():
    """Broadcast real-time updates to connected clients"""
    while True:
        try:
            # Get latest data
            patient_data = api.get_patient_monitoring_data()
            alerts = api.get_alerts_by_status()
            
            # Emit updates to all connected clients
            socketio.emit('patient_update', patient_data)
            socketio.emit('alerts_update', {
                'ongoing': alerts['ongoing'][:5],  # Latest 5 ongoing alerts
                'resolved': alerts['resolved'][:3],  # Latest 3 resolved
                'unacknowledged_count': len([a for a in alerts['ongoing'] if not a['acknowledged']])
            })
            
            time.sleep(3)  # Update every 3 seconds
        except Exception as e:
            print(f"Error in broadcast_updates: {e}")
            time.sleep(5)

# Start background thread for real-time updates
def start_background_thread():
    thread = threading.Thread(target=broadcast_updates)
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    print("=" * 50)
    print("LifeLine Integration API Server")
    print("=" * 50)
    print("Backend API available at: http://localhost:5000")
    print("WebSocket for real-time updates enabled")
    print("CORS enabled for: http://localhost:3000")
    print("=" * 50)
    
    # Start background thread
    start_background_thread()
    
    # Start the API server
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)