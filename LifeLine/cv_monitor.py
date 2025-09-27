# cv_monitor.py
import os
import time
import csv
from datetime import datetime
import math

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
import pandas as pd

# ---- CONFIG ----
CAM_INDEX = 1
SAVE_DIR = "snapshots"
os.makedirs(SAVE_DIR, exist_ok=True)

EMOTION_MT_CNN = True         # FER argument
MIN_DET_CONF = 0.5            # mediapipe face detection confidence
SAVE_INTERVAL_SEC = 5         # save image & row every N seconds
MISSING_FACE_SEC = 3          # no-face > this => missing-face alert
MOVEMENT_THRESHOLD_PX = 60    # sudden movement threshold (pixels)
EMOTION_INTERVAL_FRAMES = 5   # run heavy emotion detection every N frames

CSV_PATH = os.path.join(SAVE_DIR, "predictions.csv")
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "emotion", "score", "yaw", "pitch", "roll", "face_x", "face_y", "face_w", "face_h"])

# ---- INITIALIZE DETECTORS ----
mp_face_det = mp.solutions.face_detection.FaceDetection(min_detection_confidence=MIN_DET_CONF)


# ---- HELPERS ----
def init_camera(i=1):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"Cannot open camera index {i}, trying index 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open any camera (tried indices 0 and {})".format(i))
    return cap

def get_first_face_keypoints(face_detector, frame_bgr):
    """
    Return dict with pixel coords for right_eye, left_eye, nose, mouth and bbox (x,y,w,h).
    Uses mediapipe face_detection which returns relative_keypoints and relative_bounding_box.
    If no face -> return None.
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)
    if not results.detections:
        return None

    det = results.detections[0]
    h, w, _ = frame_bgr.shape

    # mediapipe relative_keypoints ordering (common): [right_eye, left_eye, nose_tip, mouth_center, right_ear_tragion]
    try:
        kps = det.location_data.relative_keypoints
        right_eye = (int(kps[0].x * w), int(kps[0].y * h))
        left_eye  = (int(kps[1].x * w), int(kps[1].y * h))
        nose      = (int(kps[2].x * w), int(kps[2].y * h))
        mouth     = (int(kps[3].x * w), int(kps[3].y * h))
    except Exception:
        return None

    rbox = det.location_data.relative_bounding_box
    x = max(0, int(rbox.xmin * w))
    y = max(0, int(rbox.ymin * h))
    bw = int(rbox.width * w)
    bh = int(rbox.height * h)

    return {
        "right_eye": right_eye,
        "left_eye": left_eye,
        "nose": nose,
        "mouth": mouth,
        "bbox": (x, y, bw, bh),
        "confidence": det.score[0] if det.score else None
    }

def estimate_head_pose(kp):
    """
    Simple heuristic head pose using eye line and nose position.
    Returns yaw, pitch, roll in degrees (approximate).
    - roll: tilt of eye line
    - yaw: nose horizontal offset relative to eye midpoint
    - pitch: nose vertical offset relative to eye midpoint
    These are approximate but useful for alerts / trend plotting.
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
    yaw = math.degrees(math.atan2((nx - eye_mid_x), eye_dist))  # sign: positive if nose is to right

    # pitch: vertical nose offset relative to eye midpoint, normalized
    eye_mid_y = (ry + ly) / 2.0
    pitch = math.degrees(math.atan2((eye_mid_y - ny), eye_dist))  # positive -> looking down (nose below eyes)

    return float(yaw), float(pitch), float(roll)

def detect_emotion_on_face(frame, bbox):
    """
    Crop face bbox, run DeepFace analysis, and return the dominant emotion.
    Returns (emotion_label, score) or (None, None)
    """
    x, y, w, h = bbox
    # Crop the face from the original frame
    face = frame[y:y+h, x:x+w]

    if face.size == 0:
        return None, None

    try:
        # DeepFace analyze returns a list of results, we take the first one
        # BGR=True because cv2 reads images in BGR format
        result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False, silent=True)
        
        # The result object structure is a bit different
        first_result = result[0]
        dominant_emotion = first_result['dominant_emotion']
        score = first_result['emotion'][dominant_emotion]

        # DeepFace score is 0-100, so we normalize it to 0-1
        return dominant_emotion, float(score / 100.0)
    except Exception as e:
        # DeepFace will raise an exception if it can't find a face in the cropped image
        # print(f"DeepFace error: {e}") # Optional: for debugging
        return None, None

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
        hp_text = f"Yaw:{yaw:.1f} Pitch:{pitch:.1f} Roll:{roll:.1f}"
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
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame not read, exiting")
                break

            frame_count += 1
            kp = get_first_face_keypoints(mp_face_det, frame)
            alerts = []
            headpose = (None, None, None)
            emotion = None
            score = None

            if kp is None:
                # no face detected
                if time.time() - last_seen_time > MISSING_FACE_SEC:
                    alerts.append("No face detected")
            else:
                last_seen_time = time.time()
                # head pose estimation
                yaw, pitch, roll = estimate_head_pose(kp)
                headpose = (yaw, pitch, roll)

                # sudden movement detection (use nose)
                nx, ny = kp['nose']
                if prev_nose is not None:
                    movement = math.hypot(nx - prev_nose[0], ny - prev_nose[1])
                    if movement > MOVEMENT_THRESHOLD_PX:
                        alerts.append("Sudden movement detected")
                prev_nose = (nx, ny)

                # run emotion detection every EMOTION_INTERVAL_FRAMES to save CPU
                if frame_count % EMOTION_INTERVAL_FRAMES == 0:
                    emotion, score = detect_emotion_on_face(frame, kp['bbox'])

            # save snapshot & CSV every SAVE_INTERVAL_SEC
            if time.time() - last_saved_time > SAVE_INTERVAL_SEC:
                # Only save valid data if a face was detected in the current cycle
                if kp is not None:
                    # Use previous emotion if current frame didn't run emotion detection
                    save_emotion = emotion if emotion is not None else "unknown"
                    save_score = score if score is not None else 0.0
                    save_snapshot(frame.copy(), kp['bbox'], save_emotion, save_score, headpose)
                last_saved_time = time.time()

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
