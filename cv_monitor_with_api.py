"""
CV Monitor with API Integration
===============================

Enhanced version of cv_monitor_simple.py that integrates with the API server
to provide real-time updates to the frontend.

This module extends the existing CV monitoring functionality while maintaining
backward compatibility with the original alert and prediction CSV files.
"""

import os
import time
import csv
from datetime import datetime
import math
import threading
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd

# Import the existing CV monitor functionality
from cv_monitor_simple import *

class CVMonitorWithAPI:
    def __init__(self):
        # Use existing configuration
        self.cam_index = CAM_INDEX
        self.save_dir = SAVE_DIR
        self.csv_path = CSV_PATH
        self.alerts_path = ALERTS_PATH
        
        # API integration
        self.api_server_process = None
        self.is_api_running = False
        
        # Initialize camera
        self.cap = None
        self.initialize_camera()
        
    def initialize_camera(self):
        """Initialize camera using existing function"""
        try:
            self.cap = init_camera(self.cam_index)
            print(f"Camera initialized successfully on index {self.cam_index}")
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            raise
    
    def start_api_server(self):
        """Start the API server in a separate process"""
        try:
            print("Starting API server...")
            self.api_server_process = subprocess.Popen([
                sys.executable, "api_server.py"
            ], cwd=os.getcwd())
            
            # Wait a moment for server to start
            time.sleep(3)
            self.is_api_running = True
            print("API server started at http://localhost:5000")
            
        except Exception as e:
            print(f"Failed to start API server: {e}")
            self.is_api_running = False
    
    def save_alert_with_api_notification(self, alert_type, message, severity="medium", status="ongoing"):
        """
        Save alert using existing functionality and optionally notify API clients
        """
        # Use existing save_alert function
        save_alert(alert_type, message, severity, status)
        
        # If API is running, we could add additional notification logic here
        # For now, the API server reads from the same CSV files
        print(f"Alert saved: {alert_type} - {message} [{severity}]")
    
    def run_monitoring_with_api(self):
        """
        Main monitoring loop with API integration
        Extends the existing monitoring functionality
        """
        print("=" * 60)
        print("LifeLine Patient Monitoring System with API Integration")
        print("=" * 60)
        
        # Start API server
        self.start_api_server()
        
        if not self.is_api_running:
            print("Warning: API server failed to start. Running in standalone mode.")
        
        print(f"Camera feed: {self.cam_index}")
        print(f"Snapshots saved to: {self.save_dir}")
        print(f"Predictions CSV: {self.csv_path}")
        print(f"Alerts CSV: {self.alerts_path}")
        print("=" * 60)
        print("Press 'q' to quit, 's' to take snapshot")
        print("=" * 60)
        
        # Initialize tracking variables (from original code)
        last_save = 0
        face_lost_start = None
        last_face_center = None
        snapshot_counter = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                current_time = time.time()
                
                # Face detection using existing function
                face_result = get_first_face_opencv(frame)
                
                if face_result is not None:
                    # Face detected - reset missing face timer
                    if face_lost_start is not None:
                        # Face found again - resolve any no-face alerts
                        resolve_no_face_alerts()
                        face_lost_start = None
                    
                    # Extract bbox from the dictionary returned by get_first_face_opencv
                    x, y, w, h = face_result["bbox"]
                    face_center = (x + w//2, y + h//2)
                    
                    # Movement detection (from original code)
                    if last_face_center is not None:
                        movement_dist = math.sqrt(
                            (face_center[0] - last_face_center[0])**2 + 
                            (face_center[1] - last_face_center[1])**2
                        )
                        
                        if movement_dist > MOVEMENT_THRESHOLD_PX:
                            self.save_alert_with_api_notification(
                                "sudden_movement", 
                                f"Sudden movement detected: {movement_dist:.1f}px displacement",
                                "medium"
                            )
                    
                    last_face_center = face_center
                    
                    # Simple emotion detection using existing function
                    emotion = detect_simple_emotion(frame, (x, y, w, h))
                    
                    # Head pose estimation using existing function
                    yaw, pitch, roll = estimate_head_pose(face_result)
                    
                    # Draw bounding box and info
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Emotion: {emotion}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Save data periodically
                    if current_time - last_save >= SAVE_INTERVAL_SEC:
                        save_snapshot(frame, (x, y, w, h), emotion, 0.8, (yaw, pitch, roll))
                        
                        # Also save to emotion predictions CSV for API
                        with open("emotion_predictions.csv", "a", newline="", encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                datetime.now().isoformat(),
                                emotion,
                                0.8,  # confidence
                                True  # face_detected
                            ])
                        
                        # Save snapshot
                        snapshot_path = os.path.join(self.save_dir, f"snapshot_{snapshot_counter:04d}.jpg")
                        cv2.imwrite(snapshot_path, frame)
                        snapshot_counter += 1
                        
                        last_save = current_time
                        print(f"Data saved: {emotion} | Pose: Y={yaw:.1f} P={pitch:.1f} R={roll:.1f}")
                
                else:
                    # No face detected
                    if face_lost_start is None:
                        face_lost_start = current_time
                        print("Face lost - starting timer")
                    
                    elif current_time - face_lost_start > MISSING_FACE_SEC:
                        # Generate missing face alert
                        duration = current_time - face_lost_start
                        self.save_alert_with_api_notification(
                            "no_face_detected",
                            f"No face detected for {duration:.1f} seconds",
                            "high"
                        )
                        
                        # Also save to emotion predictions CSV for API (no face detected)
                        with open("emotion_predictions.csv", "a", newline="", encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                datetime.now().isoformat(),
                                "no_face",
                                0.0,  # confidence
                                False  # face_detected
                            ])
                        
                        face_lost_start = current_time  # Reset timer to avoid spam
                    
                    # Draw "No Face" indicator
                    cv2.putText(frame, "NO FACE DETECTED", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display frame
                cv2.imshow('LifeLine Patient Monitor', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    # Manual snapshot
                    snapshot_path = os.path.join(self.save_dir, f"manual_snapshot_{int(current_time)}.jpg")
                    cv2.imwrite(snapshot_path, frame)
                    print(f"Manual snapshot saved: {snapshot_path}")
        
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Stop API server
        if self.api_server_process is not None:
            try:
                self.api_server_process.terminate()
                self.api_server_process.wait(timeout=5)
                print("API server stopped")
            except:
                self.api_server_process.kill()
                print("API server forcefully stopped")

def main():
    """Main function to start the integrated monitoring system"""
    try:
        monitor = CVMonitorWithAPI()
        monitor.run_monitoring_with_api()
    except Exception as e:
        print(f"Error starting monitoring system: {e}")
        raise

if __name__ == "__main__":
    main()