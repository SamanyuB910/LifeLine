# CV Monitor System

A real-time computer vision monitoring system that detects facial emotions and head pose using AI.

## Features

- **Real-time face detection** using MediaPipe
- **Emotion recognition** using DeepFace AI
- **Head pose estimation** (yaw, pitch, roll angles)
- **Movement detection alerts**
- **Live web dashboard** with Streamlit
- **Data logging** to CSV files
- **Snapshot saving** for analysis

## Files

- `cv_monitor.py` - Main computer vision monitoring application
- `streamlit_dashboard.py` - Web dashboard for live visualization
- `requirements.txt` - Python dependencies
- `start_system.bat` - Easy startup script (Windows)

## Installation

1. Make sure Python is installed on your system
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Manual Start
1. **Start the CV monitor:**
   ```bash
   python cv_monitor.py
   ```
   - Press 'q' to quit
   - Camera feed will show with overlays
   - Data is saved to `snapshots/` folder

2. **Start the dashboard (in a separate terminal):**
   ```bash
   streamlit run streamlit_dashboard.py
   ```
   - Open http://localhost:8501 in your browser
   - View live feed and emotion analytics

### Option 2: Automatic Start (Windows)
- Double-click `start_system.bat` to start both applications

## Configuration

Edit the constants in `cv_monitor.py`:
- `CAM_INDEX` - Camera index (default: 1, fallback to 0)
- `SAVE_INTERVAL_SEC` - How often to save snapshots (default: 5 seconds)
- `MISSING_FACE_SEC` - Alert threshold for no face detected (default: 3 seconds)
- `MOVEMENT_THRESHOLD_PX` - Movement detection sensitivity (default: 60 pixels)

## Output Files

- `snapshots/latest.jpg` - Latest camera frame
- `snapshots/predictions.csv` - All emotion and pose data
- `snapshots/*.jpg` - Timestamped snapshots

## Troubleshooting

- **Camera not found**: The system will try camera index 1, then fallback to 0
- **Import errors**: Make sure all packages are installed: `pip install -r requirements.txt`
- **DeepFace errors**: First run may take longer as it downloads AI models
- **Performance**: Emotion detection runs every 5 frames to save CPU

## Requirements

- Python 3.7+
- Webcam or camera
- Internet connection (for first-time DeepFace model download)
