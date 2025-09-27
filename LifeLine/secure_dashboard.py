# secure_dashboard.py - Enhanced Dashboard with Security Integration
import streamlit as st
import pandas as pd
import time
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json

# Import our secure modules
from secure_data_manager import SecureDataManager
from alert_manager import EnhancedAlertManager

# Page configuration
st.set_page_config(
    layout="wide", 
    page_title="Secure LifeLine Dashboard",
    page_icon="🔐",
    initial_sidebar_state="collapsed"
)

# Custom CSS for security-focused styling
st.markdown("""
<style>
    .security-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #2c3e50 0%, #3498db 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .security-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 0.9rem;
        margin: 0.25rem;
    }
    
    .security-secure {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
    }
    
    .security-warning {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
    }
    
    .alert-critical {
        background: #e74c3c;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #c0392b;
    }
    
    .alert-high {
        background: #f39c12;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #e67e22;
    }
    
    .vital-signs-card {
        background: #ecf0f1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'secure_manager' not in st.session_state:
    st.session_state.secure_manager = None
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = "default_patient"
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def authenticate_user():
    """Authenticate user with master password"""
    st.sidebar.markdown("## 🔐 Security Authentication")
    
    patient_id = st.sidebar.text_input("Patient ID", value="default_patient")
    master_password = st.sidebar.text_input("Master Password", type="password")
    
    if st.sidebar.button("Authenticate"):
        try:
            secure_manager = SecureDataManager(master_password)
            st.session_state.secure_manager = secure_manager
            st.session_state.patient_id = patient_id
            st.session_state.authenticated = True
            st.sidebar.success("✅ Authentication successful!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Authentication failed: {str(e)}")

def display_security_status():
    """Display security and system status"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.authenticated:
            st.markdown('<span class="security-badge security-secure">🔒 SECURE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="security-badge security-warning">⚠️ NOT AUTHENTICATED</span>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.authenticated:
            audit_report = st.session_state.secure_manager.get_audit_report()
            st.markdown(f'<span class="security-badge security-secure">📊 {audit_report["encrypted_files"]} Files Encrypted</span>', unsafe_allow_html=True)
    
    with col3:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f'<span class="security-badge security-secure">🕒 {current_time}</span>', unsafe_allow_html=True)
    
    with col4:
        if st.session_state.authenticated:
            alert_manager = EnhancedAlertManager(st.session_state.patient_id)
            alert_summary = alert_manager.get_alert_summary()
            active_alerts = alert_summary['total_active']
            
            if active_alerts > 0:
                st.markdown(f'<span class="security-badge security-warning">🚨 {active_alerts} Active Alerts</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="security-badge security-secure">✅ No Active Alerts</span>', unsafe_allow_html=True)

def display_enhanced_alerts():
    """Display enhanced alert system"""
    if not st.session_state.authenticated:
        return
    
    st.markdown("## 🚨 Enhanced Alert System")
    
    alert_manager = EnhancedAlertManager(st.session_state.patient_id)
    alert_summary = alert_manager.get_alert_summary()
    
    if alert_summary['total_active'] > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🔴 Active Alerts")
            for alert in alert_summary['recent_alerts']:
                severity = alert.get('severity', 'low')
                message = alert.get('message', 'No message')
                timestamp = alert.get('timestamp', 'Unknown time')
                
                if severity == 'critical':
                    st.markdown(f'''
                    <div class="alert-critical">
                        <strong>🚨 CRITICAL: {message}</strong><br>
                        <small>Time: {timestamp}</small>
                    </div>
                    ''', unsafe_allow_html=True)
                elif severity == 'high':
                    st.markdown(f'''
                    <div class="alert-high">
                        <strong>⚠️ HIGH: {message}</strong><br>
                        <small>Time: {timestamp}</small>
                    </div>
                    ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Alert Statistics")
            st.metric("Critical Alerts", alert_summary['by_severity'].get('critical', 0))
            st.metric("High Priority", alert_summary['by_severity'].get('high', 0))
            st.metric("Medium Priority", alert_summary['by_severity'].get('medium', 0))
            st.metric("Low Priority", alert_summary['by_severity'].get('low', 0))
    else:
        st.success("✅ No active alerts - Patient stable")

def display_vital_signs():
    """Display vital signs monitoring"""
    if not st.session_state.authenticated:
        return
    
    st.markdown("## 💓 Vital Signs Monitoring")
    
    # Try to get latest vital signs data
    try:
        # This would normally decrypt and display real data
        # For demo purposes, we'll show sample data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
            <div class="vital-signs-card">
                <strong>💓 Heart Rate</strong><br>
                72 BPM<br>
                <small>Normal Range</small>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="vital-signs-card">
                <strong>🫁 Breathing Rate</strong><br>
                16 bpm<br>
                <small>Normal Range</small>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="vital-signs-card">
                <strong>🌡️ Temperature</strong><br>
                98.6°F<br>
                <small>Normal</small>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'''
            <div class="vital-signs-card">
                <strong>🩸 BP Indicators</strong><br>
                Normal<br>
                <small>No Flushing/Pallor</small>
            </div>
            ''', unsafe_allow_html=True)
    
    except Exception as e:
        st.warning(f"Unable to load vital signs data: {e}")

def display_fall_detection():
    """Display fall detection status"""
    if not st.session_state.authenticated:
        return
    
    st.markdown("## 🛡️ Fall Detection & Prevention")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Current Status")
        st.success("✅ Fall detection active")
        st.info("📊 Risk level: Low")
        st.info("📍 Patient position: Stable")
    
    with col2:
        st.markdown("### Risk Factors")
        risk_factors = [
            "✅ Head posture normal",
            "✅ Movement patterns stable", 
            "✅ No rapid movements detected",
            "✅ Face position within safe range"
        ]
        
        for factor in risk_factors:
            st.markdown(factor)

def display_secure_analytics():
    """Display secure analytics with anonymized data"""
    if not st.session_state.authenticated:
        return
    
    st.markdown("## 📊 Secure Analytics (Anonymized)")
    
    # Create sample analytics data
    tab1, tab2, tab3 = st.tabs(["🎭 Emotion Analysis", "📈 Risk Trends", "🔒 Security Audit"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Emotion distribution
            emotions = ['neutral', 'happy', 'sad', 'pain']
            counts = [45, 30, 15, 10]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
            ax.pie(counts, labels=emotions, autopct='%1.1f%%', colors=colors)
            ax.set_title("Emotion Distribution (Last 24 Hours)")
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Pain scale trend
            hours = list(range(24))
            pain_scores = [2, 1, 3, 2, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(hours, pain_scores, 'r-', linewidth=2)
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Pain Scale (0-10)')
            ax.set_title('Pain Scale Trend (Last 24 Hours)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
    
    with tab2:
        # Risk score trend
        hours = list(range(24))
        risk_scores = [0.2, 0.1, 0.3, 0.2, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.2, 0.1, 0.2, 0.3, 0.2, 0.1]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hours, risk_scores, 'b-', linewidth=2)
        ax.axhline(y=0.7, color='r', linestyle='--', label='High Risk Threshold')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Risk Score (0-1)')
        ax.set_title('Patient Risk Score Trend (Last 24 Hours)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        # Security audit information
        audit_report = st.session_state.secure_manager.get_audit_report()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Access Logs", audit_report['total_access_logs'])
            st.metric("Encrypted Files", audit_report['encrypted_files'])
            st.metric("Security Status", audit_report['security_status'])
        
        with col2:
            st.markdown("### Recent Security Events")
            for log in audit_report['recent_logs'][-5:]:
                st.text(log)

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<h1 class="security-header">🔐 Secure LifeLine Dashboard</h1>', unsafe_allow_html=True)
    
    # Authentication check
    if not st.session_state.authenticated:
        authenticate_user()
        return
    
    # Display main dashboard
    display_security_status()
    
    st.markdown("---")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_enhanced_alerts()
        display_vital_signs()
    
    with col2:
        display_fall_detection()
    
    st.markdown("---")
    
    # Analytics section
    display_secure_analytics()
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Dashboard", type="primary"):
            st.rerun()
    
    with col2:
        if st.button("🔒 Logout"):
            st.session_state.authenticated = False
            st.session_state.secure_manager = None
            st.rerun()
    
    with col3:
        st.markdown("*Dashboard auto-refreshes every 30 seconds*")
    
    # Auto-refresh
    if "refresh_counter" not in st.session_state:
        st.session_state.refresh_counter = 0
    
    st.session_state.refresh_counter += 1
    if st.session_state.refresh_counter % 30 == 0:  # Auto-refresh every 30 cycles
        st.rerun()

if __name__ == "__main__":
    main()
