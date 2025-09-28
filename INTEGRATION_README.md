# LifeLine Integration Guide

This document explains how to run the integrated LifeLine Patient Monitoring System with the Next.js frontend and Python backend.

## 🏗️ System Architecture

```
┌─────────────────┐    HTTP/WebSocket    ┌──────────────────┐
│   Next.js       │ ←─────────────────→ │   Flask API      │
│   Frontend      │    Port 3000         │   Server         │
│   Dashboard     │                      │   Port 5000      │
└─────────────────┘                      └──────────────────┘
                                                   │
                                                   │ Reads/Writes
                                                   ▼
                                         ┌──────────────────┐
                                         │   CV Monitor     │
                                         │   + CSV Files    │
                                         │   (Backend)      │
                                         └──────────────────┘
```

## 🚀 Quick Start

### Option 1: Automatic Setup (Recommended)

1. **Run the setup script:**
   ```batch
   start_lifeline_system.bat
   ```

2. **Start CV monitoring:**
   ```batch
   start_cv_monitor.bat
   ```

3. **Open your browser to:**
   - Frontend Dashboard: http://localhost:3000
   - API Status: http://localhost:5000/api/system-status

### Option 2: Manual Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements_api.txt
   ```

2. **Install Frontend dependencies:**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

3. **Start the API server:**
   ```bash
   python api_server.py
   ```

4. **Start the frontend (in a new terminal):**
   ```bash
   cd frontend
   npm run dev
   ```

5. **Start CV monitoring (in a new terminal):**
   ```bash
   python cv_monitor_with_api.py
   ```

## 🧪 Testing the Integration

Run the integration test to verify everything is working:

```bash
python test_integration.py
```

This will test all API endpoints and verify the connection between frontend and backend.

## 📊 System Components

### Backend Components

1. **`api_server.py`** - Flask API server that bridges frontend and backend
   - Provides REST endpoints for patient data
   - WebSocket support for real-time updates
   - Reads from existing CSV files (alerts.csv, emotion_predictions.csv)

2. **`cv_monitor_with_api.py`** - Enhanced CV monitoring
   - Extends existing cv_monitor_simple.py
   - Integrates with API server
   - Maintains backward compatibility

3. **Existing CV System** - Your original monitoring code
   - `cv_monitor_simple.py` - Core CV monitoring
   - `streamlit_dashboard.py` - Original dashboard (still works)
   - Alert and prediction CSV files

### Frontend Components

Your Next.js frontend remains **completely unchanged**:
- Components in `frontend/components/`
- Dashboard layout and styling
- All UI components and design

## 🔗 API Endpoints

The integration provides these endpoints for the frontend:

- `GET /api/patients/patient_001` - Patient monitoring data
- `GET /api/alerts` - All alerts grouped by status
- `GET /api/alerts/ongoing` - Ongoing alerts only  
- `GET /api/alerts/resolved` - Resolved alerts only
- `GET /api/alerts/previous` - Previous alerts only
- `GET /api/system-status` - System health status
- `GET /api/video-stream-status` - Video feed status

WebSocket events:
- `patient_update` - Real-time patient data
- `alerts_update` - Real-time alert updates

## 📁 File Structure

```
LifeLine/
├── api_server.py              # NEW: Flask API server
├── cv_monitor_with_api.py     # NEW: Enhanced CV monitor
├── test_integration.py        # NEW: Integration tests
├── requirements_api.txt       # NEW: API dependencies
├── start_lifeline_system.bat  # NEW: System launcher
├── start_cv_monitor.bat       # NEW: CV monitor launcher
│
├── cv_monitor_simple.py       # EXISTING: Original CV monitor
├── streamlit_dashboard.py     # EXISTING: Original dashboard  
├── alerts.csv                 # EXISTING: Alert data
├── emotion_predictions.csv    # EXISTING: Emotion data
├── ...                        # EXISTING: Other original files
│
└── frontend/                  # UNCHANGED: Your Next.js app
    ├── .env.local             # NEW: Environment config
    ├── app/
    ├── components/
    ├── lib/
    └── ...
```

## 🛠️ Configuration

### Backend Configuration

The API server is configured in `api_server.py`:
- Port: 5000
- CORS enabled for http://localhost:3000
- WebSocket enabled
- Real-time updates every 3 seconds

### Frontend Configuration

Environment variables in `frontend/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:5000
NEXT_PUBLIC_WS_URL=http://localhost:5000
NEXT_PUBLIC_PATIENT_ID=patient_001
```

## 🔍 Troubleshooting

### Common Issues

1. **Port conflicts:**
   - Backend uses port 5000
   - Frontend uses port 3000
   - Make sure these ports are available

2. **Dependencies not installed:**
   ```bash
   pip install -r requirements_api.txt
   cd frontend && npm install
   ```

3. **API server not responding:**
   - Check if `python api_server.py` is running
   - Visit http://localhost:5000/api/system-status

4. **Frontend not connecting:**
   - Check browser console for errors
   - Verify .env.local configuration
   - Ensure CORS is enabled

### Debug Mode

Start components individually for debugging:

1. **API server with debug:**
   ```bash
   python api_server.py
   ```

2. **Frontend with debug:**
   ```bash
   cd frontend && npm run dev
   ```

3. **CV monitor with debug:**
   ```bash
   python cv_monitor_with_api.py
   ```

## 🎯 Data Flow

1. **CV Monitor** detects faces, emotions, movement → Saves to CSV files
2. **API Server** reads CSV files → Provides REST endpoints + WebSocket updates  
3. **Frontend** fetches from API → Displays real-time dashboard
4. **Real-time Updates** via WebSocket → Live data without page refresh

## 🔄 Backward Compatibility

- All existing CSV files continue to work
- Original `cv_monitor_simple.py` still functional
- Original Streamlit dashboard still available
- No changes to existing alert/prediction formats

## 📈 Next Steps

1. **Start the system** using the quick start guide
2. **Test the integration** with `python test_integration.py`
3. **Open the dashboard** at http://localhost:3000
4. **Start monitoring** with the CV system
5. **View real-time updates** in the browser

The system is now fully integrated while preserving all existing functionality!