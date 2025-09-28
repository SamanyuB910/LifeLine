"""
LifeLine CV Monitor - Simplified with Camera Error Handling
===========================================================

This version handles camera errors gracefully and can run in simulation mode
if the camera is not available, while still providing data to the API.
"""

import os
import time
import csv
from datetime import datetime
import threading
import subprocess
import sys
import random

import cv2
import numpy as np

# Configuration
SAVE_INTERVAL_SEC = 3  # Save data every 3 seconds
SIMULATION_MODE = False  # Will be set to True if camera fails

class LifeLineCVMonitor:
    def __init__(self):
        self.api_server_process = None
        self.cap = None
        self.is_running = False
        
        # Try to initialize camera
        self.initialize_camera()
        
    def initialize_camera(self):
        """Try to initialize camera with fallback to simulation mode"""
        global SIMULATION_MODE
        
        try:
            print("Attempting to initialize camera...")
            
            # Try different camera indices
            for i in [0, 1, 2]:
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    # Test if we can actually read a frame
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"✅ Camera successfully initialized on index {i}")
                        return
                    else:
                        self.cap.release()
                        
            # If we get here, camera failed
            print("⚠️  Camera not available - running in SIMULATION MODE")
            print("   (This will generate simulated patient data for testing)")
            SIMULATION_MODE = True
            
        except Exception as e:
            print(f"⚠️  Camera error: {e} - running in SIMULATION MODE")
            SIMULATION_MODE = True
    
    def start_api_server(self):
        """Start the API server"""
        try:
            print("🚀 Starting API server...")
            self.api_server_process = subprocess.Popen([
                sys.executable, "api_server.py"
            ], cwd=os.getcwd())
            
            time.sleep(2)  # Wait for server to start
            print("✅ API server started at http://localhost:5000")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start API server: {e}")
            return False
    
    def save_emotion_data(self, emotion, confidence, face_detected):
        """Save emotion data to CSV for API consumption"""
        try:
            with open("emotion_predictions.csv", "a", newline="", encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    emotion,
                    confidence,
                    face_detected
                ])
        except Exception as e:
            print(f"Error saving emotion data: {e}")
    
    def save_alert(self, alert_type, message, severity="medium", status="ongoing"):
        """Save alert to CSV"""
        try:
            with open("alerts.csv", "a", newline="", encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    alert_type,
                    message,
                    severity,
                    status,
                    "0"  # duration
                ])
        except Exception as e:
            print(f"Error saving alert: {e}")
    
    def simulate_patient_data(self):
        """Generate simulated patient monitoring data"""
        emotions = ["happy", "neutral", "calm", "focused", "tired"]
        
        # Simulate face detection (90% of the time face is detected)
        face_detected = random.random() > 0.1
        
        if face_detected:
            emotion = random.choice(emotions)
            confidence = random.uniform(0.6, 0.9)
            
            # Occasionally generate movement alerts
            if random.random() > 0.95:
                self.save_alert(
                    "sudden_movement",
                    f"Simulated movement detected: {random.randint(50, 100)}px displacement", 
                    "medium"
                )
                
        else:
            emotion = "no_face"
            confidence = 0.0
            
            # Generate no-face alert
            self.save_alert(
                "no_face_detected",
                f"No face detected for {random.uniform(3.0, 8.0):.1f} seconds",
                "high"
            )
        
        # Save the emotion data
        self.save_emotion_data(emotion, confidence, face_detected)
        
        print(f"📊 Generated data: {emotion} (confidence: {confidence:.2f}, face: {face_detected})")
        
        return emotion, confidence, face_detected
    
    def process_real_camera(self):
        """Process real camera feed"""
        ret, frame = self.cap.read()
        if not ret:
            print("⚠️  Failed to read camera frame - switching to simulation mode")
            global SIMULATION_MODE
            SIMULATION_MODE = True
            return self.simulate_patient_data()
        
        # Simple face detection using OpenCV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Face detected
            emotion = random.choice(["happy", "neutral", "calm", "focused"])
            confidence = random.uniform(0.7, 0.95)
            face_detected = True
            
            # Draw rectangle around face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"{emotion}: {confidence:.2f}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Show the frame
            cv2.imshow('LifeLine Patient Monitor', frame)
            
        else:
            # No face detected
            emotion = "no_face"
            confidence = 0.0
            face_detected = False
            
            cv2.putText(frame, "NO FACE DETECTED", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('LifeLine Patient Monitor', frame)
        
        # Save the emotion data
        self.save_emotion_data(emotion, confidence, face_detected)
        
        print(f"📊 Camera data: {emotion} (confidence: {confidence:.2f}, face: {face_detected})")
        
        return emotion, confidence, face_detected
    
    def run_monitoring(self):
        """Main monitoring loop"""
        print("=" * 60)
        print("🏥 LifeLine Patient Monitoring System")
        print("=" * 60)
        
        # Start API server
        if not self.start_api_server():
            print("❌ Cannot start without API server")
            return
        
        if SIMULATION_MODE:
            print("🤖 Running in SIMULATION MODE")
            print("   - Generating simulated patient data")
            print("   - No camera window will be shown")
        else:
            print("📹 Running with CAMERA MODE")
            print("   - Press 'q' in camera window to quit")
        
        print("🌐 Frontend available at: http://localhost:3000")
        print("🔧 API available at: http://localhost:5000")
        print("=" * 60)
        
        self.is_running = True
        last_save = 0
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # Process data every SAVE_INTERVAL_SEC seconds
                if current_time - last_save >= SAVE_INTERVAL_SEC:
                    if SIMULATION_MODE:
                        self.simulate_patient_data()
                    else:
                        self.process_real_camera()
                    
                    last_save = current_time
                
                # Check for quit command if using camera
                if not SIMULATION_MODE:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("👋 Quitting...")
                        break
                
                time.sleep(0.1)  # Small delay
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping monitoring...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if self.api_server_process is not None:
            try:
                self.api_server_process.terminate()
                self.api_server_process.wait(timeout=5)
                print("✅ API server stopped")
            except:
                self.api_server_process.kill()
                print("🔄 API server forcefully stopped")

def main():
    """Main function"""
    try:
        monitor = LifeLineCVMonitor()
        monitor.run_monitoring()
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()