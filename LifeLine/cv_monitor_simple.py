# cv_monitor_simple.py - Simplified version without heavy AI dependencies
import os
import time
import csv
from datetime import datetime
import math

import cv2
import numpy as np
import pandas as pd

# ---- CONFIG ----
CAM_INDEX = 1
SAVE_DIR = "snapshots"
os.makedirs(SAVE_DIR, exist_ok=True)

MIN_DET_CONF = 0.5            # face detection confidence
SAVE_INTERVAL_SEC = 5         # save image & row every N seconds
MISSING_FACE_SEC = 3          # no-face > this => missing-face alert
MOVEMENT_THRESHOLD_PX = 60    # sudden movement threshold (pixels)

CSV_PATH = os.path.join(SAVE_DIR, "predictions.csv")
ALERTS_PATH = os.path.join(SAVE_DIR, "alerts.csv")

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "emotion", "score", "yaw", "pitch", "roll", "face_x", "face_y", "face_w", "face_h"])

if not os.path.exists(ALERTS_PATH):
    with open(ALERTS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "alert_type", "message", "severity"])

# ---- INITIALIZE DETECTORS ----
# Use OpenCV's built-in Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---- HELPERS ----
def init_camera(i=1):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"Cannot open camera index {i}, trying index 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open any camera (tried indices 0 and {})".format(i))
    return cap

def get_first_face_opencv(frame_bgr):
    """
    Use OpenCV Haar Cascade to detect faces.
    Returns dict with bbox and estimated keypoints based on face rectangle.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None
    
    # Take the first (largest) face
    x, y, w, h = faces[0]
    
    # Estimate keypoints based on face rectangle
    # These are rough estimates for compatibility with the original code
    right_eye = (x + int(w * 0.3), y + int(h * 0.35))
    left_eye = (x + int(w * 0.7), y + int(h * 0.35))
    nose = (x + int(w * 0.5), y + int(h * 0.55))
    mouth = (x + int(w * 0.5), y + int(h * 0.75))
    
    return {
        "right_eye": right_eye,
        "left_eye": left_eye,
        "nose": nose,
        "mouth": mouth,
        "bbox": (x, y, w, h),
        "confidence": 0.8  # Fixed confidence for OpenCV detection
    }

def estimate_head_pose(kp):
    """
    Simple heuristic head pose using eye line and nose position.
    Returns yaw, pitch, roll in degrees (approximate).
    """
    rx, ry = kp['right_eye']
    lx, ly = kp['left_eye']
    nx, ny = kp['nose']

    dx = rx - lx
    dy = ry - ly
    eye_dist = math.hypot(dx, dy) + 1e-6

    # roll: tilt of the eye line
    roll = math.degrees(math.atan2(dy, dx))

    # yaw: how far nose is from eye midpoint horizontally, normalized by eye distance
    eye_mid_x = (rx + lx) / 2.0
    yaw = math.degrees(math.atan2((nx - eye_mid_x), eye_dist))

    # pitch: vertical nose offset relative to eye midpoint, normalized
    eye_mid_y = (ry + ly) / 2.0
    pitch = math.degrees(math.atan2((eye_mid_y - ny), eye_dist))

    return float(yaw), float(pitch), float(roll)

def detect_simple_emotion(frame, bbox):
    """
    Simple emotion detection based on basic image analysis.
    This is a placeholder - returns random emotion for demo purposes.
    In a real implementation, you would use a proper emotion recognition model.
    """
    x, y, w, h = bbox
    face_region = frame[y:y+h, x:x+w]
    
    if face_region.size == 0:
        return "neutral", 0.5
    
    # Convert to grayscale for analysis
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    
    # Enhanced emotion detection using facial geometry
    height, width = gray_face.shape
    
    # Analyze mouth region (bottom third of face)
    mouth_region = gray_face[int(height*0.6):int(height*0.9), int(width*0.2):int(width*0.8)]
    
    # Analyze eye region (top third of face)
    eye_region = gray_face[int(height*0.2):int(height*0.5), int(width*0.1):int(width*0.9)]
    
    if mouth_region.size == 0 or eye_region.size == 0:
        return "neutral", 0.6
    
    # Calculate brightness and contrast metrics
    avg_brightness = np.mean(gray_face)
    mouth_brightness = np.mean(mouth_region)
    eye_brightness = np.mean(eye_region)
    
    # Detect edges to find smile curves and eye patterns
    mouth_edges = cv2.Canny(mouth_region, 30, 100)
    eye_edges = cv2.Canny(eye_region, 20, 80)
    
    mouth_edge_density = np.sum(mouth_edges > 0) / mouth_edges.size
    eye_edge_density = np.sum(eye_edges > 0) / eye_edges.size
    
    # Enhanced happy detection logic
    mouth_smile_indicator = mouth_brightness > avg_brightness * 1.05  # Bright mouth area
    eye_squint_indicator = eye_edge_density > 0.05  # Squinted/smiling eyes
    overall_brightness = avg_brightness > 100  # Well-lit face
    
    # Improved emotion classification
    if mouth_smile_indicator and eye_squint_indicator:
        return "happy", 0.85
    elif mouth_smile_indicator and overall_brightness:
        return "happy", 0.75
    elif mouth_brightness > avg_brightness * 1.1:
        return "happy", 0.70
    elif avg_brightness < 70:
        return "sad", 0.65
    elif mouth_edge_density > 0.08:  # High mouth activity
        if avg_brightness > 90:
            return "surprised", 0.70
        else:
            return "angry", 0.60
    elif eye_edge_density > 0.07:  # High eye activity
        return "surprised", 0.65
    else:
        return "neutral", 0.80

def save_alert(alert_type, message, severity="medium", status="ongoing"):
    """
    Save alert to alerts CSV file for dashboard monitoring with enhanced status tracking.
    severity: "low", "medium", "high"
    status: "ongoing", "resolved", "previous"
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create alerts directory if it doesn't exist
        os.makedirs("snapshots", exist_ok=True)
        
        alert_data = {
            'timestamp': timestamp,
            'alert_type': alert_type,
            'message': message,
            'severity': severity,
            'status': status,
            'duration': 0
        }
        
        # Read existing alerts or create new dataframe
        if os.path.exists(ALERTS_PATH):
            df = pd.read_csv(ALERTS_PATH)
            df = pd.concat([df, pd.DataFrame([alert_data])], ignore_index=True)
        else:
            df = pd.DataFrame([alert_data])
        
        df.to_csv(ALERTS_PATH, index=False)
        print(f"🚨 Alert saved: {alert_type} - {message} [{status}]")
    except Exception as e:
        print(f"Error saving alert: {e}")

def update_alert_status():
    """Update alert statuses based on current conditions and timing"""
    try:
        if not os.path.exists(ALERTS_PATH):
            return
            
        df = pd.read_csv(ALERTS_PATH)
        if df.empty:
            return
            
        current_time = datetime.now()
        updated = False
        
        for idx, alert in df.iterrows():
            alert_time = datetime.strptime(alert['timestamp'], "%Y-%m-%d %H:%M:%S")
            time_diff = (current_time - alert_time).total_seconds()
            
            # Update duration for all alerts
            df.at[idx, 'duration'] = int(time_diff)
            
            # Check if alert should be moved to 'previous' (older than 2 minutes)
            if time_diff > 120 and alert['status'] != 'previous':
                df.at[idx, 'status'] = 'previous'
                updated = True
                print(f"📝 Alert moved to previous: {alert['alert_type']}")
        
        if updated:
            df.to_csv(ALERTS_PATH, index=False)
            
    except Exception as e:
        print(f"Error updating alert status: {e}")

def resolve_no_face_alerts():
    """Mark no-face alerts as resolved when face is detected"""
    try:
        if not os.path.exists(ALERTS_PATH):
            return
            
        df = pd.read_csv(ALERTS_PATH)
        if df.empty:
            return
            
        current_time = datetime.now()
        updated = False
        
        # Find ongoing no-face alerts
        for idx, alert in df.iterrows():
            if alert['alert_type'] == 'no_face_detected' and alert['status'] == 'ongoing':
                alert_time = datetime.strptime(alert['timestamp'], "%Y-%m-%d %H:%M:%S")
                duration = (current_time - alert_time).total_seconds()
                
                # Resolve the alert since face is now detected
                df.at[idx, 'status'] = 'resolved'
                df.at[idx, 'duration'] = int(duration)
                updated = True
                print(f"✅ No-face alert resolved after {duration:.1f} seconds")
        
        if updated:
            df.to_csv(ALERTS_PATH, index=False)
            
    except Exception as e:
        print(f"Error resolving no-face alerts: {e}")

def check_ongoing_no_face_alert():
    """Check if there's an ongoing no-face alert within the last 5 seconds"""
    try:
        if not os.path.exists(ALERTS_PATH):
            return False
            
        df = pd.read_csv(ALERTS_PATH)
        if df.empty:
            return False
            
        current_time = datetime.now()
        
        # Check for recent ongoing no-face alerts
        for _, alert in df.iterrows():
            if (alert['alert_type'] == 'no_face_detected' and 
                alert['status'] == 'ongoing'):
                alert_time = datetime.strptime(alert['timestamp'], "%Y-%m-%d %H:%M:%S")
                time_diff = (current_time - alert_time).total_seconds()
                
                # If alert is within 5 seconds, don't create a new one
                if time_diff <= 5:
                    return True
        
        return False
        
    except Exception as e:
        print(f"Error checking ongoing alerts: {e}")
        return False

def save_snapshot_no_face(frame, save_dir=SAVE_DIR):
    """
    Save snapshot when no face is detected to ensure dashboard stays updated.
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    fname = os.path.join(save_dir, f"{ts}_no_face.jpg")
    cv2.imwrite(fname, frame)

    # Save entry to CSV with special "no_face" emotion
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, "no_face", 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0])

    # Always update latest.jpg for dashboard refresh
    latest_img = os.path.join(save_dir, "latest.jpg")
    cv2.imwrite(latest_img, frame)

def save_snapshot(frame, bbox, emotion, score, headpose, save_dir=SAVE_DIR):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    fname = os.path.join(save_dir, f"{ts}.jpg")
    cv2.imwrite(fname, frame)

    x, y, w, h = bbox
    yaw, pitch, roll = headpose
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, emotion, score, yaw, pitch, roll, x, y, w, h])

    # also keep a latest copy for dashboard
    latest_img = os.path.join(save_dir, "latest.jpg")
    cv2.imwrite(latest_img, frame)

def overlay_info(frame, kp, emotion, score, headpose, alerts):
    """
    Draw bbox/keypoints + text overlays and alerts onto frame.
    """
    if kp is not None:
        x, y, w, h = kp['bbox']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)
        for name in ['left_eye', 'right_eye', 'nose', 'mouth']:
            cx, cy = kp[name]
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

    # emotion text
    text = f"Emotion: {emotion} ({score:.2f})" if (emotion and score) else "Emotion: -"
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    if headpose:
        yaw, pitch, roll = headpose
        if None not in (yaw, pitch, roll):
            hp_text = f"Yaw:{yaw:.1f} Pitch:{pitch:.1f} Roll:{roll:.1f}"
        else:
            hp_text = "Yaw: - Pitch: - Roll: -"
        cv2.putText(frame, hp_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 2)

    # alerts
    y0 = 75
    for a in alerts:
        cv2.putText(frame, f"ALERT: {a}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        y0 += 25

def run_monitor():
    cap = init_camera(CAM_INDEX)
    prev_nose = None
    last_seen_time = time.time()
    last_saved_time = 0
    last_alert_time = 0
    frame_count = 0
    consecutive_sad_angry = 0  # Track agitation
    last_no_face_alert = 0

    print("CV Monitor started successfully!")
    print("Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame not read, exiting")
                break

            frame_count += 1
            kp = get_first_face_opencv(frame)
            alerts = []
            headpose = (None, None, None)
            emotion = None
            score = None
            current_time = time.time()

            # Always update alert statuses
            update_alert_status()

            if kp is None:
                # No face detected
                time_since_face = current_time - last_seen_time
                if time_since_face > MISSING_FACE_SEC:
                    alerts.append("No face detected")
                    # Send alert to dashboard only if no ongoing alert in last 5 seconds
                    if not check_ongoing_no_face_alert():
                        save_alert("no_face_detected", 
                                 f"Patient not visible for {time_since_face:.1f} seconds", 
                                 "high", "ongoing")
                        last_no_face_alert = current_time
                
                # Save snapshot even when no face to keep dashboard updated
                save_snapshot_no_face(frame)
            else:
                # Face detected - resolve any ongoing no-face alerts
                resolve_no_face_alerts()
                last_seen_time = current_time
                
                # head pose estimation
                yaw, pitch, roll = estimate_head_pose(kp)
                headpose = (yaw, pitch, roll)

                # Enhanced emotion detection
                emotion, score = detect_simple_emotion(frame, kp['bbox'])

                # Track agitation (consecutive sad/angry emotions)
                if emotion in ['sad', 'angry'] and score > 0.6:
                    consecutive_sad_angry += 1
                    if consecutive_sad_angry >= 15:  # 15 consecutive frames (~3 seconds)
                        alerts.append("Patient agitation detected")
                        if current_time - last_alert_time > 30:  # Alert max once every 30 seconds
                            save_alert("agitation_detected", 
                                     f"Sustained {emotion} expression detected", 
                                     "medium", "ongoing")
                            last_alert_time = current_time
                        consecutive_sad_angry = 10  # Reset but keep some history
                else:
                    consecutive_sad_angry = max(0, consecutive_sad_angry - 1)  # Decay counter

                # sudden movement detection (use nose)
                nx, ny = kp['nose']
                if prev_nose is not None:
                    movement = math.hypot(nx - prev_nose[0], ny - prev_nose[1])
                    if movement > MOVEMENT_THRESHOLD_PX:
                        alerts.append("Sudden movement detected")
                        if current_time - last_alert_time > 5:  # Alert max once every 5 seconds
                            save_alert("movement_alert", 
                                     f"Sudden movement: {movement:.1f}px displacement", 
                                     "low", "ongoing")
                            last_alert_time = current_time
                prev_nose = (nx, ny)

            # save snapshot & CSV every SAVE_INTERVAL_SEC
            if current_time - last_saved_time > SAVE_INTERVAL_SEC:
                if kp is not None:
                    # Normal case: face detected, save full data
                    save_emotion = emotion if emotion is not None else "unknown"
                    save_score = score if score is not None else 0.0
                    save_snapshot(frame.copy(), kp['bbox'], save_emotion, save_score, headpose)
                else:
                    # No face detected: save frame with special "no_face" entry
                    # This ensures dashboard gets updated even when no face is present
                    save_snapshot_no_face(frame.copy())
                last_saved_time = current_time

            overlay_info(frame, kp, emotion, score, headpose, alerts)
            cv2.imshow("CV Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_monitor()
