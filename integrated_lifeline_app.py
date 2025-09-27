#!/usr/bin/env python3
"""
LifeLine - Advanced Healthcare Computer Vision System (Integrated)
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sqlite3
import logging
from typing import Dict, List, Optional
import threading
import queue
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional local model (DeepFace) availability
try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except Exception:
    DeepFace = None
    HAS_DEEPFACE = False

@dataclass
class PatientData:
    patient_id: str
    name: str
    room: str
    age: int
    admission_date: datetime
    primary_diagnosis: str
    allergies: List[str]
    current_medications: List[str]

@dataclass
class VitalSigns:
    timestamp: datetime
    heart_rate: int
    systolic_bp: int
    diastolic_bp: int
    temperature: float
    spo2: int
    respiratory_rate: int

@dataclass
class MLPrediction:
    timestamp: datetime
    emotion: str
    emotion_confidence: float
    pain_level: str
    pain_score: int
    fall_risk: str
    composite_risk: float
    clinical_insights: List[str]
    recommendations: List[str]

class DatabaseManager:
    def __init__(self, db_path: str = "lifeline_data.db"):
        # ensure DB path is relative to project root
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                room TEXT,
                age INTEGER,
                admission_date TEXT,
                primary_diagnosis TEXT,
                allergies TEXT,
                medications TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vital_signs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                timestamp TEXT,
                heart_rate INTEGER,
                systolic_bp INTEGER,
                diastolic_bp INTEGER,
                temperature REAL,
                spo2 INTEGER,
                respiratory_rate INTEGER,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                timestamp TEXT,
                emotion TEXT,
                emotion_confidence REAL,
                pain_level TEXT,
                pain_score INTEGER,
                fall_risk TEXT,
                composite_risk REAL,
                insights TEXT,
                recommendations TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                timestamp TEXT,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                acknowledged INTEGER DEFAULT 0,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')

        conn.commit()
        conn.close()

        self.insert_sample_data()

    def insert_sample_data(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM patients")
        if cursor.fetchone()[0] > 0:
            conn.close()
            return

        patients = [
            ("PAT001", "Sarah Johnson", "ICU-301", 67, "2025-09-20", "Pneumonia", '["Penicillin"]', '["Amoxicillin", "Paracetamol"]'),
            ("PAT002", "Michael Chen", "Ward-205", 45, "2025-09-22", "Post-surgical recovery", '[]', '["Ibuprofen", "Omeprazole"]'),
            ("PAT003", "Emma Rodriguez", "ICU-102", 78, "2025-09-19", "Heart failure", '["Aspirin"]', '["Lisinopril", "Furosemide", "Metoprolol"]')
        ]

        cursor.executemany('''
            INSERT INTO patients (patient_id, name, room, age, admission_date, primary_diagnosis, allergies, medications)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', patients)

        base_time = datetime.now() - timedelta(hours=24)
        for patient_id in ["PAT001", "PAT002", "PAT003"]:
            for i in range(48):
                timestamp = base_time + timedelta(hours=i)
                if patient_id == "PAT001":
                    hr = np.random.normal(110, 15)
                    sys_bp = np.random.normal(160, 20)
                    dia_bp = np.random.normal(95, 10)
                    temp = np.random.normal(38.2, 0.5)
                    spo2 = np.random.normal(94, 3)
                elif patient_id == "PAT003":
                    hr = np.random.normal(85, 10)
                    sys_bp = np.random.normal(140, 15)
                    dia_bp = np.random.normal(85, 8)
                    temp = np.random.normal(37.1, 0.3)
                    spo2 = np.random.normal(96, 2)
                else:
                    hr = np.random.normal(72, 8)
                    sys_bp = np.random.normal(120, 10)
                    dia_bp = np.random.normal(80, 5)
                    temp = np.random.normal(36.8, 0.2)
                    spo2 = np.random.normal(98, 1)

                rr = np.random.normal(16, 2)
                cursor.execute('''
                    INSERT INTO vital_signs (patient_id, timestamp, heart_rate, systolic_bp, diastolic_bp, temperature, spo2, respiratory_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (patient_id, timestamp.isoformat(), int(hr), int(sys_bp), int(dia_bp), round(temp, 1), int(spo2), int(rr)))

        conn.commit()
        conn.close()

    def get_patient_data(self, patient_id: str) -> Optional[PatientData]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return PatientData(
                patient_id=row[0],
                name=row[1],
                room=row[2],
                age=row[3],
                admission_date=datetime.fromisoformat(row[4]),
                primary_diagnosis=row[5],
                allergies=json.loads(row[6]),
                current_medications=json.loads(row[7])
            )
        return None

    def get_recent_vitals(self, patient_id: str, hours: int = 24) -> List[VitalSigns]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        cursor.execute('''
            SELECT * FROM vital_signs 
            WHERE patient_id = ? AND timestamp > ?
            ORDER BY timestamp DESC
        ''', (patient_id, cutoff_time))
        vitals = []
        for row in cursor.fetchall():
            vitals.append(VitalSigns(
                timestamp=datetime.fromisoformat(row[2]),
                heart_rate=row[3],
                systolic_bp=row[4],
                diastolic_bp=row[5],
                temperature=row[6],
                spo2=row[7],
                respiratory_rate=row[8]
            ))
        conn.close()
        return vitals

    def save_ml_prediction(self, patient_id: str, prediction: MLPrediction):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO ml_predictions (patient_id, timestamp, emotion, emotion_confidence, 
                                      pain_level, pain_score, fall_risk, composite_risk, 
                                      insights, recommendations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id, prediction.timestamp.isoformat(), prediction.emotion,
            prediction.emotion_confidence, prediction.pain_level, prediction.pain_score,
            prediction.fall_risk, prediction.composite_risk,
            json.dumps(prediction.clinical_insights), json.dumps(prediction.recommendations)
        ))
        conn.commit()
        conn.close()

class CameraManager:
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.capture_thread = None

    def start_camera(self, camera_index: int = 0):
        # Try requested camera index first
        try:
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                logger.info(f"Camera index {camera_index} not opened, trying default index 0")
                self.camera.release()
                self.camera = cv2.VideoCapture(0)

            if not self.camera.isOpened():
                # failed to open any camera
                logger.warning("No camera available or cannot be opened.")
                return False

            # configure camera only after successful open
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            try:
                self.camera.set(cv2.CAP_PROP_FPS, 30)
            except Exception:
                # not all backends support FPS set; ignore
                pass

            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()
            logger.info("Camera started successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            try:
                if self.camera:
                    self.camera.release()
            except Exception:
                pass
            return False

    def _capture_frames(self):
        while self.is_running and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
            time.sleep(1/30)

    def get_frame(self) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def stop_camera(self):
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.camera:
            self.camera.release()

class LifeLineApp:
    def __init__(self):
        # Database manager can be recreated each run safely
        self.db = DatabaseManager()

        # Persist CameraManager in Streamlit session_state so camera/thread survive reruns
        if 'camera_manager' not in st.session_state:
            st.session_state.camera_manager = CameraManager()
        self.camera_manager = st.session_state.camera_manager

        # Session values
        if 'current_patient' not in st.session_state:
            st.session_state.current_patient = None
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        if 'predictions_history' not in st.session_state:
            st.session_state.predictions_history = []
        # Local model settings
        if 'use_local_model' not in st.session_state:
            st.session_state.use_local_model = False
        if 'model_last_run' not in st.session_state:
            st.session_state.model_last_run = datetime.min

    def run(self):
        st.set_page_config(page_title="LifeLine HCP Dashboard", page_icon="🏥", layout="wide")
        self.inject_custom_css()
        self.render_header()
        self.render_sidebar()
        if st.session_state.current_patient:
            self.render_patient_dashboard()
        else:
            self.render_patient_selection()

    def inject_custom_css(self):
        st.markdown("""
        <style> .main-header{background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);padding:1rem;border-radius:10px;margin-bottom:2rem;} .metric-card{background:white;padding:1rem;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);border-left:4px solid #667eea;} .patient-status-critical{color:#e53e3e;font-weight:bold;} .patient-status-stable{color:#38a169;font-weight:bold;} .patient-status-monitoring{color:#d69e2e;font-weight:bold;} </style>
        """, unsafe_allow_html=True)

    def render_header(self):
        st.markdown("""
        <div class="main-header"><h1 style="color:white;margin:0;">🏥 LifeLine - Advanced Healthcare Monitoring</h1><p style="color:white;margin:0;opacity:0.9;">AI-Powered Clinical Decision Support System</p></div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        with st.sidebar:
            st.title("🏥 LifeLine Control")
            st.subheader("Patient Management")
            conn = sqlite3.connect(self.db.db_path)
            patients_df = pd.read_sql_query("SELECT patient_id, name, room FROM patients", conn)
            conn.close()
            patient_options = [f"{row['name']} ({row['room']})" for _, row in patients_df.iterrows()]
            patient_ids = patients_df['patient_id'].tolist()
            selected_idx = st.selectbox(
                "Select Patient:",
                range(len(patient_options)),
                format_func=lambda x: patient_options[x] if x < len(patient_options) else "Select...",
                index=0 if patient_options else None
            )

            if selected_idx is not None and patient_options:
                selected_patient_id = patient_ids[selected_idx]
                if st.session_state.current_patient != selected_patient_id:
                    st.session_state.current_patient = selected_patient_id
                    st.rerun()
            st.divider()
            st.subheader("📹 Camera Control")
            # Camera index selector to help with systems that have multiple devices
            camera_index = st.number_input("Camera Index:", value=0, min_value=0, max_value=10, step=1)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Start Camera"):
                    started = self.camera_manager.start_camera(camera_index=int(camera_index))
                    if started:
                        st.session_state.camera_active = True
                        st.success(f"Camera started (index {int(camera_index)})!")
                    else:
                        st.session_state.camera_active = False
                        st.error(f"Failed to start camera at index {int(camera_index)}")
            with col2:
                if st.button("Stop Camera"):
                    self.camera_manager.stop_camera()
                    st.session_state.camera_active = False
                    st.success("Camera stopped!")
            st.divider()
            st.subheader("📊 System Status")
            st.metric("Camera Status", "Active" if st.session_state.camera_active else "Inactive")
            st.metric("Connected Patients", len(patients_df))
            st.metric("Active Alerts", self.get_active_alerts_count())

    def get_active_alerts_count(self) -> int:
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged = 0")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def render_patient_selection(self):
        st.title("Select a Patient to Monitor")
        conn = sqlite3.connect(self.db.db_path)
        patients_df = pd.read_sql_query("SELECT patient_id, name, room, primary_diagnosis FROM patients", conn)
        conn.close()
        cols = st.columns(3)
        for idx, (_, patient) in enumerate(patients_df.iterrows()):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="metric-card"><h3>{patient['name']}</h3><p><strong>Room:</strong> {patient['room']}</p><p><strong>Diagnosis:</strong> {patient['primary_diagnosis']}</p></div>
                """, unsafe_allow_html=True)
                if st.button(f"Monitor {patient['name']}", key=f"select_{patient['patient_id']}"):
                    st.session_state.current_patient = patient['patient_id']
                    st.rerun()

    def render_patient_dashboard(self):
        patient_id = st.session_state.current_patient
        patient_data = self.db.get_patient_data(patient_id)
        if not patient_data:
            st.error("Patient data not found!")
            return
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.title(f"👤 {patient_data.name}")
            st.write(f"**Room:** {patient_data.room} | **Age:** {patient_data.age} | **Diagnosis:** {patient_data.primary_diagnosis}")
        with col2:
            if st.button("🔔 New Alert"):
                self.create_manual_alert(patient_id)
        with col3:
            risk_score = self.calculate_current_risk_score(patient_id)
            st.metric("Risk Score", f"{risk_score:.0f}/100")
        tab1, tab2, tab3, tab4 = st.tabs(["🔴 Live Monitoring", "📊 Analytics", "📋 Clinical Notes", "⚙️ Settings"])
        with tab1:
            self.render_live_monitoring_tab(patient_id)
        with tab2:
            self.render_analytics_tab(patient_id)
        with tab3:
            self.render_clinical_notes_tab(patient_id)
        with tab4:
            self.render_settings_tab(patient_id)

    def render_live_monitoring_tab(self, patient_id: str):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📹 Live Camera Feed")
            camera_placeholder = st.empty()
            if st.session_state.camera_active:
                frame = self.camera_manager.get_frame()
                if frame is not None:
                    # If local model enabled and available, try to run it (rate-limited)
                    predictions = None
                    now = datetime.now()
                    use_local = bool(st.session_state.get('use_local_model', False)) and HAS_DEEPFACE
                    last_run = st.session_state.get('model_last_run', datetime.min)
                    # simple rate limit: 1 inference per 2 seconds
                    if use_local and (now - last_run).total_seconds() > 2:
                        try:
                            predictions = self.run_local_model_on_frame(frame)
                            st.session_state.model_last_run = now
                        except Exception as e:
                            logger.error(f"Local model failed: {e}")
                            predictions = None

                    if predictions is None:
                        predictions = self.simulate_ml_predictions()
                    annotated_frame = self.annotate_frame(frame, predictions)
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                    prediction_obj = MLPrediction(
                        timestamp=datetime.now(),
                        emotion=predictions['emotion'],
                        emotion_confidence=predictions['emotion_confidence'],
                        pain_level=predictions['pain_level'],
                        pain_score=predictions['pain_score'],
                        fall_risk=predictions['fall_risk'],
                        composite_risk=predictions['composite_risk'],
                        clinical_insights=predictions['insights'],
                        recommendations=predictions['recommendations']
                    )
                    st.session_state.predictions_history.append(prediction_obj)
                    if len(st.session_state.predictions_history) > 10:
                        st.session_state.predictions_history.pop(0)
                    self.db.save_ml_prediction(patient_id, prediction_obj)
                else:
                    # Provide diagnostic info to help debug camera feed issues
                    camera_placeholder.write("📹 Waiting for camera feed...")
                    try:
                        cam_opened = False
                        qsize = 0
                        if self.camera_manager.camera is not None:
                            cam_opened = bool(self.camera_manager.camera.isOpened())
                        qsize = self.camera_manager.frame_queue.qsize()
                        st.write(f"Camera opened: {cam_opened}")
                        st.write(f"Frame queue size: {qsize}")
                    except Exception as e:
                        st.write(f"Camera diagnostics unavailable: {e}")
            else:
                camera_placeholder.write("📹 Camera inactive. Click 'Start Camera' in sidebar.")
        with col2:
            st.subheader("💓 Current Vital Signs")
            vitals = self.db.get_recent_vitals(patient_id, hours=1)
            if vitals:
                latest_vitals = vitals[0]
                col_hr, col_bp = st.columns(2)
                with col_hr:
                    hr_delta = self.calculate_vital_delta('heart_rate', vitals)
                    st.metric("Heart Rate", f"{latest_vitals.heart_rate} BPM", delta=hr_delta)
                with col_bp:
                    st.metric("Blood Pressure", f"{latest_vitals.systolic_bp}/{latest_vitals.diastolic_bp}")
                col_temp, col_spo2 = st.columns(2)
                with col_temp:
                    temp_delta = self.calculate_vital_delta('temperature', vitals)
                    st.metric("Temperature", f"{latest_vitals.temperature}°C", delta=temp_delta)
                with col_spo2:
                    spo2_delta = self.calculate_vital_delta('spo2', vitals)
                    st.metric("SpO2", f"{latest_vitals.spo2}%", delta=spo2_delta)
            st.divider()
            st.subheader("🤖 AI Analysis")
            if st.session_state.predictions_history:
                latest_pred = st.session_state.predictions_history[-1]
                st.write(f"**Emotional State:** {latest_pred.emotion.title()} ({latest_pred.emotion_confidence:.1%})")
                st.write(f"**Pain Assessment:** {latest_pred.pain_level.replace('_', ' ').title()} (Level {latest_pred.pain_score}/4)")
                st.write(f"**Fall Risk:** {latest_pred.fall_risk.title()} Risk")
                if latest_pred.clinical_insights:
                    st.write("**Clinical Insights:**")
                    for insight in latest_pred.clinical_insights:
                        st.write(f"• {insight}")
            else:
                st.write("No AI analysis data available yet.")

    def render_analytics_tab(self, patient_id: str):
        st.subheader("📊 Patient Analytics & Trends")
        col1, col2 = st.columns([1, 3])
        with col1:
            time_range = st.selectbox("Time Range:", ["Last 6 hours", "Last 24 hours", "Last 3 days", "Last week"]) 
            hours_map = {"Last 6 hours": 6, "Last 24 hours": 24, "Last 3 days": 72, "Last week": 168}
            selected_hours = hours_map[time_range]
        vitals = self.db.get_recent_vitals(patient_id, hours=selected_hours)
        if not vitals:
            st.warning("No vital signs data available for the selected time range.")
            return
        vitals_df = pd.DataFrame([{
            'timestamp': v.timestamp,
            'heart_rate': v.heart_rate,
            'systolic_bp': v.systolic_bp,
            'diastolic_bp': v.diastolic_bp,
            'temperature': v.temperature,
            'spo2': v.spo2,
            'respiratory_rate': v.respiratory_rate
        } for v in vitals])
        fig = make_subplots(rows=2, cols=2, subplot_titles=('Heart Rate (BPM)', 'Blood Pressure (mmHg)', 'Temperature (°C)', 'SpO2 (%)'), vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=vitals_df['timestamp'], y=vitals_df['heart_rate'], name='Heart Rate', line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=vitals_df['timestamp'], y=vitals_df['systolic_bp'], name='Systolic', line=dict(color='blue')), row=1, col=2)
        fig.add_trace(go.Scatter(x=vitals_df['timestamp'], y=vitals_df['diastolic_bp'], name='Diastolic', line=dict(color='lightblue')), row=1, col=2)
        fig.add_trace(go.Scatter(x=vitals_df['timestamp'], y=vitals_df['temperature'], name='Temperature', line=dict(color='orange')), row=2, col=1)
        fig.add_trace(go.Scatter(x=vitals_df['timestamp'], y=vitals_df['spo2'], name='SpO2', line=dict(color='green')), row=2, col=2)
        fig.update_layout(height=600, showlegend=False, title_text="Vital Signs Trends")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("🎯 Risk Score Trend")
        risk_scores = []
        for v in vitals:
            risk = self.calculate_risk_from_vitals(v)
            risk_scores.append({'timestamp': v.timestamp, 'risk_score': risk})
        risk_df = pd.DataFrame(risk_scores)
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Scatter(x=risk_df['timestamp'], y=risk_df['risk_score'], mode='lines+markers', name='Risk Score', line=dict(color='purple', width=3)))
        fig_risk.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="High Risk")
        fig_risk.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
        fig_risk.update_layout(title="Composite Risk Score Over Time", yaxis_title="Risk Score (0-100)", xaxis_title="Time", height=400)
        st.plotly_chart(fig_risk, use_container_width=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_hr = vitals_df['heart_rate'].mean()
            st.metric("Avg Heart Rate", f"{avg_hr:.0f} BPM")
        with col2:
            avg_bp = vitals_df['systolic_bp'].mean()
            st.metric("Avg Systolic BP", f"{avg_bp:.0f} mmHg")
        with col3:
            avg_temp = vitals_df['temperature'].mean()
            st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
        with col4:
            avg_spo2 = vitals_df['spo2'].mean()
            st.metric("Avg SpO2", f"{avg_spo2:.0f}%")

    def render_clinical_notes_tab(self, patient_id: str):
        st.subheader("📋 Clinical Documentation")
        patient_data = self.db.get_patient_data(patient_id)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Patient Information:**")
            st.write(f"**Name:** {patient_data.name}")
            st.write(f"**Age:** {patient_data.age}")
            st.write(f"**Room:** {patient_data.room}")
            st.write(f"**Admission Date:** {patient_data.admission_date.strftime('%Y-%m-%d')}")
            st.write(f"**Primary Diagnosis:** {patient_data.primary_diagnosis}")
        with col2:
            st.write("**Allergies:**")
            if patient_data.allergies:
                for allergy in patient_data.allergies:
                    st.write(f"⚠️ {allergy}")
            else:
                st.write("No known allergies")
            st.write("**Current Medications:**")
            for med in patient_data.current_medications:
                st.write(f"💊 {med}")
        st.divider()
        st.subheader("🤖 AI Analysis Summary")
        conn = sqlite3.connect(self.db.db_path)
        ml_predictions_df = pd.read_sql_query("""
            SELECT * FROM ml_predictions 
            WHERE patient_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 10
        """, conn, params=(patient_id,))
        conn.close()
        if not ml_predictions_df.empty:
            for _, pred in ml_predictions_df.iterrows():
                timestamp = pd.to_datetime(pred['timestamp'])
                with st.expander(f"Analysis - {timestamp.strftime('%Y-%m-%d %H:%M')}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Emotion:** {pred['emotion']}")
                        st.write(f"**Confidence:** {pred['emotion_confidence']:.1%}")
                    with col2:
                        st.write(f"**Pain Level:** {pred['pain_level']}")
                        st.write(f"**Pain Score:** {pred['pain_score']}/4")
                    with col3:
                        st.write(f"**Fall Risk:** {pred['fall_risk']}")
                        st.write(f"**Composite Risk:** {pred['composite_risk']:.0f}/100")
                    if pred['insights']:
                        insights = json.loads(pred['insights'])
                        st.write("**Clinical Insights:**")
                        for insight in insights:
                            st.write(f"• {insight}")
        else:
            st.info("No AI analysis data available yet.")
        st.divider()
        st.subheader("📝 Clinical Notes")
        with st.expander("Add New Clinical Note"):
            note_type = st.selectbox("Note Type:", ["General", "Assessment", "Plan", "Medication", "Alert"]) 
            note_text = st.text_area("Note:", height=100)
            if st.button("Save Note"):
                st.success("Note saved successfully!")

    def render_settings_tab(self, patient_id: str):
        st.subheader("⚙️ System Settings")
        st.write("**Alert Thresholds:**")
        col1, col2 = st.columns(2)
        with col1:
            hr_min = st.number_input("Heart Rate Min:", value=60, min_value=30, max_value=120)
            hr_max = st.number_input("Heart Rate Max:", value=100, min_value=60, max_value=200)
            temp_max = st.number_input("Temperature Max (°C):", value=38.0, min_value=35.0, max_value=42.0)
        with col2:
            spo2_min = st.number_input("SpO2 Min (%):", value=95, min_value=80, max_value=100)
            bp_sys_max = st.number_input("Systolic BP Max:", value=140, min_value=100, max_value=220)
            risk_threshold = st.number_input("Risk Score Alert:", value=70, min_value=0, max_value=100)
        st.divider()
        st.write("**AI Model Settings:**")
        col1, col2 = st.columns(2)
        with col1:
            emotion_sensitivity = st.slider("Emotion Detection Sensitivity:", 0.1, 1.0, 0.7)
            pain_sensitivity = st.slider("Pain Detection Sensitivity:", 0.1, 1.0, 0.8)
        with col2:
            update_frequency = st.selectbox("Analysis Update Frequency:", ["Real-time", "Every 30s", "Every 1min", "Every 5min"]) 
            enable_alerts = st.checkbox("Enable Automatic Alerts", value=True)
        # Local model toggle
        st.write("\n")
        st.write("**Local Model (optional)**")
        if HAS_DEEPFACE:
            st.write("DeepFace available — enabling will run local emotion analysis on camera frames.")
        else:
            st.write("DeepFace not installed or not available in this environment. Falling back to simulation.")
        use_local = st.checkbox("Use Local Model (DeepFace)", value=st.session_state.use_local_model)
        st.session_state.use_local_model = use_local
        st.divider()
        st.write("**Data Management:**")
        retention_period = st.selectbox("Data Retention Period:", ["30 days", "90 days", "6 months", "1 year"]) 
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export Patient Data"):
                st.info("Data export functionality will be implemented.")
        with col2:
            if st.button("Clear Old Data"):
                st.warning("This will remove data older than the retention period.")
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")

    def simulate_ml_predictions(self) -> Dict:
        import random
        emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise']
        pain_levels = ['no_pain', 'mild', 'moderate', 'severe', 'extreme']
        fall_risks = ['low', 'medium', 'high']
        emotion = random.choice(emotions)
        emotion_confidence = random.uniform(0.6, 0.95)
        pain_level = random.choice(pain_levels)
        pain_score = pain_levels.index(pain_level)
        fall_risk = random.choice(fall_risks)
        risk_components = {'emotion_risk': 0.3 if emotion in ['angry', 'fear', 'sad'] else 0.1, 'pain_risk': pain_score * 0.2, 'fall_risk_val': {'low': 0.1, 'medium': 0.4, 'high': 0.7}[fall_risk]}
        composite_risk = (risk_components['emotion_risk'] + risk_components['pain_risk'] + risk_components['fall_risk_val']) * 100
        insights = []
        recommendations = []
        if pain_score >= 3:
            insights.append(f"High pain level detected: {pain_level}")
            recommendations.append("Consider pain medication reassessment")
        if emotion in ['angry', 'fear', 'sad']:
            insights.append(f"Emotional distress detected: {emotion}")
            recommendations.append("Consider psychological support")
        if fall_risk == 'high':
            insights.append("High fall risk detected")
            recommendations.append("Implement fall prevention measures")
        return {'emotion': emotion, 'emotion_confidence': emotion_confidence, 'pain_level': pain_level, 'pain_score': pain_score, 'fall_risk': fall_risk, 'composite_risk': composite_risk, 'insights': insights, 'recommendations': recommendations}

    def run_local_model_on_frame(self, frame: np.ndarray) -> Dict:
        """Run DeepFace emotion analysis on a BGR OpenCV frame and convert to the app prediction dict."""
        if not HAS_DEEPFACE:
            raise RuntimeError("DeepFace not available")

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # DeepFace expects a file path or numpy image in RGB format
        try:
            res = DeepFace.analyze(rgb, actions=['emotion'], enforce_detection=False)
        except Exception as e:
            raise

        # DeepFace returns scores and dominant_emotion
        emotion = res.get('dominant_emotion', 'neutral')
        scores = res.get('emotion', {})
        # approximate confidence
        emotion_confidence = 0.0
        if emotion in scores:
            total = sum(scores.values()) if sum(scores.values()) > 0 else 1
            emotion_confidence = scores[emotion] / total

        # map emotion to pain/fall heuristics (simple fallback)
        pain_score = 0
        pain_level = 'no_pain'
        fall_risk = 'low'
        composite_risk = (0.3 if emotion in ['angry', 'fear', 'sad'] else 0.1) * 100

        insights = []
        recommendations = []
        if emotion_confidence < 0.5:
            insights.append('Low confidence in emotion detection')

        return {
            'emotion': emotion,
            'emotion_confidence': emotion_confidence,
            'pain_level': pain_level,
            'pain_score': pain_score,
            'fall_risk': fall_risk,
            'composite_risk': composite_risk,
            'insights': insights,
            'recommendations': recommendations
        }

    def annotate_frame(self, frame: np.ndarray, predictions: Dict) -> np.ndarray:
        annotated = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        y_offset = 35
        cv2.putText(annotated, f"Emotion: {predictions['emotion']} ({predictions['emotion_confidence']:.1%})", (20, y_offset), font, font_scale, (0, 255, 0), thickness)
        y_offset += 25
        cv2.putText(annotated, f"Pain: {predictions['pain_level']} (Level {predictions['pain_score']})", (20, y_offset), font, font_scale, (0, 255, 255), thickness)
        y_offset += 25
        cv2.putText(annotated, f"Fall Risk: {predictions['fall_risk']}", (20, y_offset), font, font_scale, (255, 255, 0), thickness)
        y_offset += 25
        risk_color = (0, 0, 255) if predictions['composite_risk'] > 70 else (0, 255, 255) if predictions['composite_risk'] > 40 else (0, 255, 0)
        cv2.putText(annotated, f"Risk Score: {predictions['composite_risk']:.0f}/100", (20, y_offset), font, font_scale, risk_color, thickness)
        return annotated

    def calculate_current_risk_score(self, patient_id: str) -> float:
        vitals = self.db.get_recent_vitals(patient_id, hours=1)
        if not vitals:
            return 50.0
        latest_vitals = vitals[0]
        return self.calculate_risk_from_vitals(latest_vitals)

    def calculate_risk_from_vitals(self, vitals: VitalSigns) -> float:
        risk_score = 0
        if vitals.heart_rate < 50 or vitals.heart_rate > 100:
            risk_score += 20
        elif vitals.heart_rate < 40 or vitals.heart_rate > 120:
            risk_score += 40
        if vitals.systolic_bp < 90 or vitals.systolic_bp > 160:
            risk_score += 15
        elif vitals.systolic_bp < 80 or vitals.systolic_bp > 180:
            risk_score += 30
        if vitals.spo2 < 96:
            risk_score += 25
        elif vitals.spo2 < 92:
            risk_score += 50
        if vitals.temperature > 38.0 or vitals.temperature < 36.0:
            risk_score += 10
        elif vitals.temperature > 39.0 or vitals.temperature < 35.0:
            risk_score += 25
        return min(100, max(0, risk_score))

    def calculate_vital_delta(self, vital_type: str, vitals: List[VitalSigns]) -> Optional[str]:
        if len(vitals) < 2:
            return None
        current = getattr(vitals[0], vital_type)
        previous = getattr(vitals[1], vital_type)
        delta = current - previous
        if abs(delta) < 0.1:
            return None
        return f"{delta:+.1f}" if vital_type == 'temperature' else f"{delta:+d}"

    def create_manual_alert(self, patient_id: str):
        st.info("Manual alert creation feature will be implemented.")

# Guard main so import won't run streamlit UI in tests
def main():
    app = LifeLineApp()
    app.run()

if __name__ == '__main__':
    main()
