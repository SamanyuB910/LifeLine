# streamlit_dashboard.py - Enhanced UI (No Plotly Required)
import streamlit as st
import pandas as pd
import time
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Page configuration with custom styling
st.set_page_config(
    layout="wide", 
    page_title="CV Monitor Dashboard",
    page_icon="📹",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    
    .status-active {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .emotion-display {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .happy { background: linear-gradient(135deg, #ffeaa7, #fdcb6e); }
    .sad { background: linear-gradient(135deg, #74b9ff, #0984e3); }
    .neutral { background: linear-gradient(135deg, #ddd, #bbb); }
    .angry { background: linear-gradient(135deg, #ff7675, #e17055); }
    .surprised { background: linear-gradient(135deg, #fd79a8, #e84393); }
    .fear { background: linear-gradient(135deg, #6c5ce7, #a29bfe); }
    .disgust { background: linear-gradient(135deg, #00b894, #00cec9); }
    
    .section-divider {
        border-top: 2px solid #e9ecef;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

SNAP_DIR = "snapshots"
IMAGE_PATH = os.path.join(SNAP_DIR, "latest.jpg")
CSV_PATH = os.path.join(SNAP_DIR, "predictions.csv")
ALERTS_PATH = os.path.join(SNAP_DIR, "alerts.csv")

# Header with emoji and styling
st.markdown('<h1 class="main-header">📹 CV Monitor Dashboard</h1>', unsafe_allow_html=True)

# Status indicator
col_status1, col_status2, col_status3, col_status4 = st.columns(4)

with col_status1:
    if os.path.exists(IMAGE_PATH):
        st.markdown('<span class="status-badge status-active">🟢 Camera Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-warning">🟡 Camera Waiting</span>', unsafe_allow_html=True)

with col_status2:
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        if not df.empty:
            st.markdown('<span class="status-badge status-active">📊 Data Recording</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-warning">📊 No Data</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-warning">📊 No Data File</span>', unsafe_allow_html=True)

with col_status3:
    current_time = datetime.now().strftime("%H:%M:%S")
    st.markdown(f'<span class="status-badge status-active">🕒 {current_time}</span>', unsafe_allow_html=True)

with col_status4:
    # Alert status
    if os.path.exists(ALERTS_PATH):
        alerts_df = pd.read_csv(ALERTS_PATH)
        if not alerts_df.empty:
            recent_alerts = alerts_df.tail(5)
            high_alerts = recent_alerts[recent_alerts['severity'] == 'high']
            if not high_alerts.empty:
                st.markdown('<span class="status-badge" style="background-color: #f8d7da; color: #721c24;">🚨 Active Alerts</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-active">✅ No Alerts</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-active">✅ No Alerts</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-active">✅ No Alerts</span>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 📸 Live Camera Feed")
    # show latest image if exists
    if os.path.exists(IMAGE_PATH):
        img = Image.open(IMAGE_PATH)
        st.image(img, caption="Latest frame from camera", use_container_width=True)
    else:
        st.info("🔄 Waiting for camera feed...")

with col2:
    st.markdown("### 🎭 Current Analysis")
    
    # show latest prediction
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        if not df.empty:
            latest = df.iloc[-1]
            
            # Emotion display with color coding
            emotion = latest['emotion']
            score = latest['score']
            
            # Handle special "no_face" case
            if emotion == "no_face":
                st.markdown(f'''
                <div class="emotion-display" style="background: linear-gradient(135deg, #ff7675, #d63031); color: white;">
                    🚫 NO FACE DETECTED<br>
                    <small>Patient not visible</small>
                </div>
                ''', unsafe_allow_html=True)
            else:
                emotion_class = emotion.lower() if emotion.lower() in ['happy', 'sad', 'neutral', 'angry', 'surprised', 'fear', 'disgust'] else 'neutral'
                
                st.markdown(f'''
                <div class="emotion-display {emotion_class}">
                    🎭 {emotion.upper()}<br>
                    <small>Confidence: {score:.1%}</small>
                </div>
                ''', unsafe_allow_html=True)
            
            # Metrics in cards
            st.markdown(f'''
            <div class="metric-card">
                <strong>📅 Timestamp:</strong><br>
                {latest['timestamp']}
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown(f'''
            <div class="metric-card">
                <strong>🧭 Head Pose:</strong><br>
                🔄 Yaw: {latest['yaw']:.1f}°<br>
                ⬆️ Pitch: {latest['pitch']:.1f}°<br>
                🔀 Roll: {latest['roll']:.1f}°
            </div>
            ''', unsafe_allow_html=True)
            
            # Enhanced Statistics with face detection awareness
            emotion_data = df[df['emotion'] != 'no_face']
            total_face_readings = len(emotion_data)
            no_face_readings = len(df[df['emotion'] == 'no_face'])
            total_readings = len(df)
            
            st.markdown(f'''
            <div class="metric-card">
                <strong>📊 Enhanced Statistics:</strong><br>
                📈 Total Records: {total_readings}<br>
                👤 Face Detected: {total_face_readings}<br>
                🚫 No Face: {no_face_readings}<br>
                📊 Detection Rate: {(total_face_readings/total_readings*100):.1f}%<br>
                {'🎯 Face Position: (' + str(int(latest["face_x"])) + ', ' + str(int(latest["face_y"])) + ')' if latest["emotion"] != "no_face" else '❌ Face Position: Not Available'}<br>
                {'📏 Face Size: ' + str(int(latest["face_w"])) + '×' + str(int(latest["face_h"])) + 'px' if latest["emotion"] != "no_face" else '📏 Face Size: Not Available'}
            </div>
            ''', unsafe_allow_html=True)
            
            # Enhanced Real-time Alerts Panel with categorization
            st.markdown("### 🚨 Live Alerts System")
            
            if os.path.exists(ALERTS_PATH):
                alerts_df = pd.read_csv(ALERTS_PATH)
                if not alerts_df.empty:
                    # Separate alerts by status
                    ongoing_alerts = alerts_df[alerts_df['status'] == 'ongoing'].sort_values('timestamp', ascending=False)
                    resolved_alerts = alerts_df[alerts_df['status'] == 'resolved'].sort_values('timestamp', ascending=False).head(3)
                    previous_alerts = alerts_df[alerts_df['status'] == 'previous'].sort_values('timestamp', ascending=False).head(3)
                    
                    # Create two columns for ongoing and historical alerts
                    col_ongoing, col_historical = st.columns([1, 1])
                    
                    with col_ongoing:
                        st.markdown("#### 🔴 Ongoing Alerts")
                        if not ongoing_alerts.empty:
                            for _, alert in ongoing_alerts.iterrows():
                                severity_color = {
                                    'high': '#ff7675',
                                    'medium': '#fdcb6e', 
                                    'low': '#74b9ff'
                                }.get(alert['severity'], '#ddd')
                                
                                severity_icon = {
                                    'high': '🚨',
                                    'medium': '⚠️',
                                    'low': 'ℹ️'
                                }.get(alert['severity'], '📝')
                                
                                # Calculate duration
                                alert_time = datetime.strptime(alert['timestamp'], "%Y-%m-%d %H:%M:%S")
                                duration = (datetime.now() - alert_time).total_seconds()
                                duration_str = f"{int(duration)}s" if duration < 60 else f"{int(duration/60)}m {int(duration%60)}s"
                                
                                st.markdown(f'''
                                <div style="background: {severity_color}; padding: 0.5rem; border-radius: 5px; margin: 0.25rem 0; color: white; border-left: 4px solid white;">
                                    <strong>{severity_icon} {alert['alert_type'].upper().replace('_', ' ')}</strong><br>
                                    {alert['message']}<br>
                                    <small>⏰ Duration: {duration_str} | 🕐 Started: {alert['timestamp']}</small>
                                </div>
                                ''', unsafe_allow_html=True)
                        else:
                            st.success("✅ No ongoing alerts - Patient stable")
                    
                    with col_historical:
                        st.markdown("#### 📋 Recent History")
                        
                        # Show recently resolved alerts
                        if not resolved_alerts.empty:
                            st.markdown("**🟢 Recently Resolved:**")
                            for _, alert in resolved_alerts.iterrows():
                                duration_str = f"{alert['duration']}s" if alert['duration'] < 60 else f"{int(alert['duration']/60)}m {int(alert['duration']%60)}s"
                                st.markdown(f'''
                                <div style="background: #00b894; padding: 0.3rem; border-radius: 3px; margin: 0.2rem 0; color: white; opacity: 0.8;">
                                    <small><strong>✅ {alert['alert_type'].upper().replace('_', ' ')}</strong><br>
                                    Duration: {duration_str} | Resolved: {alert['timestamp']}</small>
                                </div>
                                ''', unsafe_allow_html=True)
                        
                        # Show previous alerts (older than 2 minutes)
                        if not previous_alerts.empty:
                            st.markdown("**📜 Previous Alerts:**")
                            for _, alert in previous_alerts.iterrows():
                                st.markdown(f'''
                                <div style="background: #636e72; padding: 0.3rem; border-radius: 3px; margin: 0.2rem 0; color: white; opacity: 0.7;">
                                    <small><strong>📝 {alert['alert_type'].upper().replace('_', ' ')}</strong><br>
                                    {alert['timestamp']}</small>
                                </div>
                                ''', unsafe_allow_html=True)
                        
                        if resolved_alerts.empty and previous_alerts.empty:
                            st.info("📝 No recent alert history")
                            
                    # Alert statistics
                    st.markdown("---")
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    
                    with col_stats1:
                        ongoing_count = len(ongoing_alerts)
                        st.metric("🔴 Ongoing", ongoing_count, delta="Active alerts")
                    
                    with col_stats2:
                        resolved_count = len(resolved_alerts)
                        st.metric("🟢 Resolved", resolved_count, delta="Last hour")
                    
                    with col_stats3:
                        total_today = len(alerts_df[alerts_df['timestamp'].str.contains(datetime.now().strftime("%Y-%m-%d"))])
                        st.metric("📊 Today", total_today, delta="Total alerts")
                        
                else:
                    st.success("✅ No alerts recorded - System monitoring active")
            else:
                st.info("📝 Alert system initializing...")
            
        else:
            st.warning("⏳ No predictions available yet")
    else:
        st.error("❌ No data file found")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Analytics section
st.markdown("### 📈 Analytics & Trends")

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    if not df.empty:
        # Create tabs for different analytics
        tab1, tab2, tab3 = st.tabs(["🎭 Emotion Analysis", "📊 Confidence Trends", "🧭 Head Pose"])
        
        with tab1:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Emotion distribution pie chart using matplotlib (exclude no_face)
                emotion_data = df[df['emotion'] != 'no_face']
                if not emotion_data.empty:
                    emotion_counts = emotion_data['emotion'].value_counts()
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = ['#ffeaa7', '#74b9ff', '#ddd', '#ff7675', '#fd79a8', '#6c5ce7', '#00b894']
                    wedges, texts, autotexts = ax.pie(
                        emotion_counts.values, 
                        labels=emotion_counts.index,
                        autopct='%1.1f%%',
                        colors=colors[:len(emotion_counts)],
                        startangle=90
                    )
                    ax.set_title("Emotion Distribution (Face Detected)", fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("No emotion data available (no faces detected)")
            
            with col_chart2:
                # Recent emotions bar chart using Streamlit (exclude no_face)
                recent_data = df.tail(20)
                recent_emotions = recent_data[recent_data['emotion'] != 'no_face']['emotion'].value_counts()
                if not recent_emotions.empty:
                    st.markdown("**Recent Emotions (Last 20)**")
                    st.bar_chart(recent_emotions)
                else:
                    st.info("No recent emotion data (no faces detected)")
        
        with tab2:
            # Confidence over time using matplotlib (exclude no_face)
            emotion_data = df[df['emotion'] != 'no_face'].tail(50).copy()
            
            if not emotion_data.empty:
                emotion_data['index'] = range(len(emotion_data))
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot different emotions with different colors
                emotions = emotion_data['emotion'].unique()
                colors = {'happy': '#ffeaa7', 'sad': '#74b9ff', 'neutral': '#ddd', 
                         'angry': '#ff7675', 'surprised': '#fd79a8', 'fear': '#6c5ce7', 'disgust': '#00b894'}
                
                for emotion in emotions:
                    emotion_subset = emotion_data[emotion_data['emotion'] == emotion]
                    ax.scatter(emotion_subset['index'], emotion_subset['score'], 
                              label=emotion, c=colors.get(emotion, '#333'), s=50, alpha=0.7)
                
                ax.plot(emotion_data['index'], emotion_data['score'], 'k-', alpha=0.3, linewidth=1)
                ax.set_xlabel('Reading Number')
                ax.set_ylabel('Confidence Score')
                ax.set_title('Confidence Scores Over Time (Face Detected Only)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No confidence data available (no faces detected)")
        
        with tab3:
            # Head pose visualization using matplotlib (exclude no_face)
            pose_data = df[df['emotion'] != 'no_face'].tail(30).copy()
            
            if not pose_data.empty:
                pose_data['reading'] = range(len(pose_data))
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(pose_data['reading'], pose_data['yaw'], 'r-', label='Yaw', linewidth=2)
                ax.plot(pose_data['reading'], pose_data['pitch'], 'g-', label='Pitch', linewidth=2)
                ax.plot(pose_data['reading'], pose_data['roll'], 'b-', label='Roll', linewidth=2)
                
                ax.set_xlabel('Reading Number')
                ax.set_ylabel('Angle (degrees)')
                ax.set_title('Head Pose Angles Over Time (Face Detected Only)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No head pose data available (no faces detected)")
            
    else:
        st.info("📊 No data available for analytics yet")
else:
    st.error("❌ Data file not found")

# Footer with refresh info
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

col_footer1, col_footer2 = st.columns(2)
with col_footer1:
    if st.button("🔄 Refresh Dashboard", type="primary"):
        st.rerun()

with col_footer2:
    st.markdown("*Dashboard auto-refreshes periodically*")

# Auto-refresh mechanism (reduced frequency for better performance)
if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0

st.session_state.refresh_counter += 1
if st.session_state.refresh_counter % 200 == 0:  # Auto-refresh every 200 cycles
    st.rerun()
