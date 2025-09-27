# LifeLine Backend

This folder contains a lightweight Flask + SocketIO backend for the LifeLine project.

Requirements

- Python 3.8+
- See top-level `requirements.txt` for packages. Install with:

```bash
python -m pip install -r requirements.txt
```

Run

- From project root (Windows PowerShell):

```powershell
& ".\.venv\Scripts\Activate.ps1"; .\run_backend.bat
```

Notes

- The backend tries to use `deepface` if available; if DeepFace or TensorFlow are not installed, the server will run but emotion analysis will return `unknown`.
- The server uses Flask-SocketIO with the `threading` async mode to avoid requiring `eventlet` or `gevent`.
