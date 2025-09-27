"""LifeLine backend server (Flask + SocketIO)

This adapts the user's provided server into a runnable module inside the LifeLine project.
It uses the threading async_mode for Flask-SocketIO to avoid requiring eventlet/gevent.
"""
from __future__ import annotations

import os
import time
import threading
import queue
import logging
from datetime import datetime
from collections import deque
from typing import Dict

from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit

import cv2
import mediapipe as mp
try:
    from deepface import DeepFace
except Exception:
    DeepFace = None

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'))
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)
# Use threading async mode to avoid adding eventlet/gevent as a hard dependency
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# MediaPipe initialization
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Global variables for system state
class SystemState:
    def __init__(self):
        self.active_patients: Dict[str, dict] = {}
        self.alerts = deque(maxlen=100)
        self.video_feed = None
        self.is_monitoring = False
        self.frame_queue = queue.Queue(maxsize=10)

system_state = SystemState()


class PatientMonitor:
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.emotion_buffer = deque(maxlen=30)
        self.movement_buffer = deque(maxlen=30)
        self.head_pose_buffer = deque(maxlen=30)
        self.last_face_time = time.time()
        self.alert_cooldown = {}
        self.risk_score = 0

    def analyze_emotion(self, frame, face_bbox):
        if DeepFace is None:
            return 'unknown', {}
        try:
            x, y, w, h = face_bbox
            face_img = frame[y:y+h, x:x+w]
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, silent=True)
            dominant_emotion = result[0]['dominant_emotion']
            emotion_scores = result[0]['emotion']
            self.emotion_buffer.append({'emotion': dominant_emotion, 'scores': emotion_scores, 'timestamp': datetime.now()})
            return dominant_emotion, emotion_scores
        except Exception as e:
            logger.debug(f"DeepFace analyze failed: {e}")
            return 'unknown', {}

    def calculate_head_pose(self, landmarks):
        # simplified head pose estimation
        try:
            nose_tip = landmarks[1]
            chin = landmarks[152]
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            yaw = np.arctan2(right_eye.x - left_eye.x, right_eye.z - left_eye.z) * 180 / np.pi
            pitch = np.arctan2(chin.y - nose_tip.y, chin.z - nose_tip.z) * 180 / np.pi
            roll = np.arctan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x) * 180 / np.pi
            self.head_pose_buffer.append({'yaw': yaw, 'pitch': pitch, 'roll': roll, 'timestamp': datetime.now()})
            return yaw, pitch, roll
        except Exception:
            return 0, 0, 0

    def detect_movement(self, current_bbox):
        if len(self.movement_buffer) > 0:
            last_bbox = self.movement_buffer[-1]['bbox']
            movement = np.sqrt((current_bbox[0] - last_bbox[0])**2 + (current_bbox[1] - last_bbox[1])**2)
            self.movement_buffer.append({'bbox': current_bbox, 'movement': movement, 'timestamp': datetime.now()})
            if movement > 50:
                self._trigger_alert('movement', 'Significant movement detected')
            return movement
        self.movement_buffer.append({'bbox': current_bbox, 'movement': 0, 'timestamp': datetime.now()})
        return 0

    def _trigger_alert(self, alert_type, message):
        cooldown_key = f"{alert_type}_{message}"
        current_time = time.time()
        if cooldown_key in self.alert_cooldown:
            if current_time - self.alert_cooldown[cooldown_key] < 30:
                return
        self.alert_cooldown[cooldown_key] = current_time
        alert = {'id': int(current_time * 1000), 'type': 'warning' if alert_type == 'movement' else 'critical', 'message': message, 'patient': f"Patient {self.patient_id}", 'time': datetime.now().isoformat(), 'alert_type': alert_type}
        system_state.alerts.append(alert)
        socketio.emit('new_alert', alert)

    def calculate_risk_score(self):
        factors = {'emotion_distress': 0, 'movement_abnormal': 0, 'pose_unusual': 0, 'no_face_detection': 0}
        if len(self.emotion_buffer) > 0:
            recent_emotions = [e['emotion'] for e in list(self.emotion_buffer)[-10:]]
            distress_ratio = sum(1 for e in recent_emotions if e in ['angry', 'fear', 'sad']) / len(recent_emotions)
            factors['emotion_distress'] = distress_ratio * 30
        if len(self.movement_buffer) > 5:
            recent_movement = [m['movement'] for m in list(self.movement_buffer)[-5:]]
            avg_movement = np.mean(recent_movement)
            if avg_movement > 30:
                factors['movement_abnormal'] = min(30, avg_movement / 2)
        if len(self.head_pose_buffer) > 5:
            recent_poses = list(self.head_pose_buffer)[-5:]
            avg_yaw = np.mean([p['yaw'] for p in recent_poses])
            avg_pitch = np.mean([p['pitch'] for p in recent_poses])
            if abs(avg_yaw) > 30 or abs(avg_pitch) > 30:
                factors['pose_unusual'] = 20
        if time.time() - self.last_face_time > 10:
            factors['no_face_detection'] = 20
        self.risk_score = min(100, sum(factors.values()))
        return self.risk_score


class VideoProcessor:
    def __init__(self):
        self.cap = None
        self.face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.monitors: Dict[str, PatientMonitor] = {}

    def start_capture(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return self.cap.isOpened()

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_detection = self.face_detection.process(rgb_frame)
        results_mesh = self.face_mesh.process(rgb_frame)
        analysis_data = {'faces_detected': 0, 'patients': []}
        if results_detection and results_detection.detections:
            for idx, detection in enumerate(results_detection.detections):
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)
                patient_id = f"patient_{idx+1}"
                if patient_id not in self.monitors:
                    self.monitors[patient_id] = PatientMonitor(patient_id)
                monitor = self.monitors[patient_id]
                monitor.last_face_time = time.time()
                emotion, emotion_scores = monitor.analyze_emotion(frame, (x, y, box_w, box_h))
                movement = monitor.detect_movement((x, y, box_w, box_h))
                yaw, pitch, roll = 0, 0, 0
                if results_mesh and results_mesh.multi_face_landmarks:
                    landmarks = results_mesh.multi_face_landmarks[0].landmark
                    yaw, pitch, roll = monitor.calculate_head_pose(landmarks)
                risk_score = monitor.calculate_risk_score()
                cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0) if risk_score < 50 else (0, 165, 255), 2)
                cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Risk: {risk_score:.0f}%", (x, y + box_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if risk_score < 50 else (0, 0, 255), 1)
                patient_data = {'id': patient_id, 'emotion': emotion, 'emotion_scores': emotion_scores, 'movement': movement, 'head_pose': {'yaw': yaw, 'pitch': pitch, 'roll': roll}, 'risk_score': risk_score, 'bbox': [x, y, box_w, box_h]}
                analysis_data['patients'].append(patient_data)
                analysis_data['faces_detected'] += 1
        return frame, analysis_data

    def generate_frames(self):
        while system_state.is_monitoring:
            if self.cap and self.cap.isOpened():
                success, frame = self.cap.read()
                if success:
                    processed_frame, analysis_data = self.process_frame(frame)
                    socketio.emit('frame_analysis', analysis_data)
                    ret, buffer = cv2.imencode('.jpg', processed_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.1)


video_processor = VideoProcessor()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/patients')
def get_patients():
    patients = []
    for patient_id, monitor in video_processor.monitors.items():
        patients.append({'id': patient_id, 'name': f"Patient {patient_id.split('_')[1]}", 'room': f"ICU-20{patient_id.split('_')[1]}", 'status': 'critical' if monitor.risk_score > 70 else 'stable' if monitor.risk_score < 40 else 'monitoring', 'risk_score': monitor.risk_score, 'last_update': datetime.now().isoformat(), 'emotion': monitor.emotion_buffer[-1]['emotion'] if monitor.emotion_buffer else 'unknown', 'movement': 'normal' if len(monitor.movement_buffer) == 0 or monitor.movement_buffer[-1]['movement'] < 30 else 'elevated'})
    return jsonify(patients)


@app.route('/api/alerts')
def get_alerts():
    return jsonify(list(system_state.alerts))


@app.route('/api/start_monitoring', methods=['POST'])
def start_monitoring():
    if not system_state.is_monitoring:
        camera_index = request.json.get('camera_index', 0)
        if video_processor.start_capture(camera_index):
            system_state.is_monitoring = True
            thread = threading.Thread(target=video_stream_thread)
            thread.daemon = True
            thread.start()
            return jsonify({'success': True, 'message': 'Monitoring started'})
        else:
            return jsonify({'error': 'Failed to start camera'}), 500
    return jsonify({'message': 'Already monitoring'})


@app.route('/api/stop_monitoring', methods=['POST'])
def stop_monitoring():
    system_state.is_monitoring = False
    if video_processor.cap:
        video_processor.cap.release()
    return jsonify({'success': True, 'message': 'Monitoring stopped'})


@app.route('/api/video_feed')
def video_feed():
    return Response(video_processor.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to LifeLine system'})


@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")


def video_stream_thread():
    while system_state.is_monitoring:
        try:
            if video_processor.cap and video_processor.cap.isOpened():
                success, frame = video_processor.cap.read()
                if success:
                    _, analysis_data = video_processor.process_frame(frame)
                    for patient_data in analysis_data['patients']:
                        patient_id = patient_data['id']
                        system_state.active_patients[patient_id] = patient_data
                    socketio.emit('system_update', {'active_patients': len(system_state.active_patients), 'latest_alerts': list(system_state.alerts)[-5:], 'timestamp': datetime.now().isoformat()})
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Video processing error: {e}")


def initialize_ml_models():
    logger.info("Initializing ML models...")
    logger.info("ML models ready")


@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'monitoring': system_state.is_monitoring, 'active_patients': len(system_state.active_patients), 'uptime': datetime.now().isoformat()})


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(__file__))
    os.makedirs(os.path.join(base_dir, 'snapshots'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'templates'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'static'), exist_ok=True)
    # lightweight template
    with open(os.path.join(base_dir, 'templates', 'index.html'), 'w', encoding='utf-8') as f:
        f.write('<html><body><h3>LifeLine Backend Running</h3><p>Visit /api/patients or connect via SocketIO.</p></body></html>')
    initialize_ml_models()
    logger.info('Starting LifeLine backend on http://localhost:5000')
    # Allow Werkzeug for local development/demo environments.
    # flask-socketio >=5 may raise a RuntimeError by default; pass allow_unsafe_werkzeug=True
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
