"""
Test script for the enhanced medical-grade monitoring system
"""

import cv2
import time
from models.data_models import PatientProfile
from models.monitor import MedicalGradeMonitor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_medical_monitor():
    # Initialize test patient
    test_patient = PatientProfile(
        patient_id="TEST001",
        name="Test Patient",
        room="ICU-1",
        age=45,
        medical_history=["hypertension"],
        monitoring_protocol="standard"
    )
    
    # Initialize the enhanced monitor
    monitor = MedicalGradeMonitor(test_patient)
    logger.info("Initialized medical-grade monitor")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return
    
    logger.info("Starting medical-grade monitoring test...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                break
            
            # Process frame with medical analysis
            processed_frame, analysis = monitor.process_frame(frame)
            
            # Log medical metrics
            if analysis.face_detected:
                logger.info(f"Risk Score: {analysis.risk_score}% | "
                          f"Pain Score: {monitor.pain_score:.1f} | "
                          f"GCS: {monitor.glasgow_coma_scale} | "
                          f"RASS: {monitor.rass_score}")
                
                # Log insights if any
                if analysis.insights:
                    for insight in analysis.insights:
                        logger.info(f"Medical Insight ({insight['priority']}): {insight['title']} - {insight['message']}")
            
            # Display the processed frame
            cv2.imshow('Medical-Grade Monitoring Test', processed_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Test terminated by user")
                break
                
            # Short sleep to prevent high CPU usage
            time.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Error during monitoring: {e}")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Test completed - Resources released")

if __name__ == "__main__":
    test_medical_monitor()