# test_cv_simple.py - Test the simplified CV monitor without camera
import os
import time
import csv
from datetime import datetime
import cv2
import numpy as np
import pandas as pd

# Test if OpenCV can load the face cascade
print("Testing OpenCV face detection...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("❌ ERROR: Could not load face cascade classifier")
else:
    print("✅ Face cascade loaded successfully")

# Test camera availability
print("\nTesting camera access...")
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✅ Camera 0 is available")
        ret, frame = cap.read()
        if ret:
            print("✅ Can read frames from camera")
            h, w = frame.shape[:2]
            print(f"✅ Frame size: {w}x{h}")
        else:
            print("❌ Cannot read frames from camera")
        cap.release()
    else:
        print("❌ Camera 0 is not available")
        # Try camera 1
        cap = cv2.VideoCapture(1)
        if cap.isOpened():
            print("✅ Camera 1 is available")
            cap.release()
        else:
            print("❌ Camera 1 is also not available")
except Exception as e:
    print(f"❌ Error accessing camera: {e}")

# Test directory creation
print("\nTesting file operations...")
SAVE_DIR = "snapshots"
try:
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("✅ Snapshots directory created/exists")
    
    # Test CSV writing
    CSV_PATH = os.path.join(SAVE_DIR, "test_predictions.csv")
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "emotion", "score", "yaw", "pitch", "roll", "face_x", "face_y", "face_w", "face_h"])
        writer.writerow(["20250902T120000", "happy", "0.85", "2.3", "-1.2", "0.5", "120", "80", "150", "180"])
    print("✅ CSV file operations work")
    
    # Clean up test file
    os.remove(CSV_PATH)
    
except Exception as e:
    print(f"❌ Error with file operations: {e}")

# Test pandas
print("\nTesting pandas...")
try:
    df = pd.DataFrame({'test': [1, 2, 3]})
    print("✅ Pandas works")
except Exception as e:
    print(f"❌ Pandas error: {e}")

# Test numpy
print("\nTesting numpy...")
try:
    arr = np.array([1, 2, 3])
    print("✅ NumPy works")
except Exception as e:
    print(f"❌ NumPy error: {e}")

print("\n=== TEST SUMMARY ===")
print("The simplified CV monitor should work if:")
print("1. ✅ Face cascade loaded")
print("2. ✅ At least one camera is available") 
print("3. ✅ File operations work")
print("4. ✅ Pandas and NumPy work")
print("\nIf camera is not available, the program will show an error but the code structure is correct.")
