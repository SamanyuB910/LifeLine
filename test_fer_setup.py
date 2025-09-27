#!/usr/bin/env python3
"""
Simple FER Training Script
=========================
Basic facial expression recognition using scikit-learn.
"""

print("🚀 Starting FER Training...")

try:
    import os
    import zipfile
    import numpy as np
    from PIL import Image
    import sklearn
    print("✅ All imports successful!")
    print(f"📦 Scikit-learn version: {sklearn.__version__}")
    
    # Check if fer.zip exists
    if os.path.exists("datasets/fer.zip"):
        print("✅ Found datasets/fer.zip")
    else:
        print("❌ datasets/fer.zip not found")
        
    print("🎉 Basic setup complete!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")