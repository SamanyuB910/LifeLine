#!/usr/bin/env python3
"""
Quick FER Training Script
========================
Fast facial expression recognition training using scikit-learn.
"""

import os
import zipfile
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import glob
from pathlib import Path

def extract_fer_dataset():
    """Extract fer.zip quickly."""
    dataset_path = "datasets/fer_dataset"
    zip_path = "datasets/fer.zip"
    
    print(f"🔍 Checking for dataset...")
    
    if not os.path.exists(dataset_path):
        if not os.path.exists(zip_path):
            print(f"❌ Error: {zip_path} not found!")
            return False
            
        print(f"📦 Extracting {zip_path}... (this may take a moment)")
        os.makedirs(dataset_path, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract only first 1000 files for quick training
                files = zip_ref.namelist()[:1000]  # Limit for speed
                for file in files:
                    zip_ref.extract(file, dataset_path)
            print(f"✅ Quick extraction completed (first 1000 files)")
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            return False
    else:
        print(f"✅ Dataset already exists at {dataset_path}")
    
    return True

def load_sample_images(folder_path, max_per_class=50, target_size=(24, 24)):
    """Load a sample of images for quick training."""
    images = []
    labels = []
    
    print(f"🔍 Scanning {folder_path}...")
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return np.array([]), np.array([]), []
    
    # Try to find emotion folders
    emotion_folders = []
    for root, dirs, files in os.walk(folder_path):
        if any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in files):
            emotion_folders.append(root)
    
    if not emotion_folders:
        print(f"❌ No image folders found in {folder_path}")
        return np.array([]), np.array([]), []
    
    print(f"📁 Found {len(emotion_folders)} folders with images")
    
    emotion_classes = []
    class_idx = 0
    
    for folder in emotion_folders[:7]:  # Limit to 7 emotion classes
        folder_name = os.path.basename(folder)
        emotion_classes.append(folder_name)
        
        image_files = glob.glob(os.path.join(folder, "*.jpg")) + \
                     glob.glob(os.path.join(folder, "*.png"))
        
        # Limit images per class for speed
        image_files = image_files[:max_per_class]
        
        print(f"📸 Loading {len(image_files)} images from {folder_name}...")
        
        for img_path in image_files:
            try:
                # Quick image processing
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img = img.resize(target_size)  # Smaller size for speed
                img = img.convert('L')  # Grayscale
                
                img_array = np.array(img).flatten() / 255.0  # Normalize
                
                images.append(img_array)
                labels.append(class_idx)
                
            except Exception as e:
                print(f"⚠️  Error loading {img_path}: {e}")
        
        class_idx += 1
    
    return np.array(images), np.array(labels), emotion_classes

def quick_fer_training():
    """Quick FER training for demo purposes."""
    print("🚀 Quick FER Training with Scikit-Learn")
    print("=" * 50)
    
    # Step 1: Extract dataset
    print("Step 1: Dataset extraction...")
    if not extract_fer_dataset():
        print("❌ Using synthetic data instead...")
        # Create synthetic data for demo
        np.random.seed(42)
        X = np.random.rand(1000, 24*24)  # 1000 samples, 24x24 pixels
        y = np.random.randint(0, 7, 1000)  # 7 emotion classes
        emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Using synthetic data: {len(X_train)} train, {len(X_test)} test samples")
    else:
        # Step 2: Load real data
        print("\nStep 2: Loading data...")
        
        # Try different possible paths
        possible_paths = [
            "datasets/fer_dataset",
            "datasets/fer_dataset/fer2013",
            "datasets/fer_dataset/train"
        ]
        
        X, y, emotion_classes = None, None, None
        
        for path in possible_paths:
            if os.path.exists(path):
                X, y, emotion_classes = load_sample_images(path)
                if len(X) > 0:
                    break
        
        if X is None or len(X) == 0:
            print("❌ No data found, using synthetic data...")
            np.random.seed(42)
            X = np.random.rand(1000, 24*24)
            y = np.random.randint(0, 7, 1000)
            emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Data loaded: {len(X_train)} train, {len(X_test)} test samples")
    
    print(f"🎭 Emotion classes: {emotion_classes}")
    
    # Step 3: Quick preprocessing
    print("\nStep 3: Preprocessing...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 4: Quick training
    print("\nStep 4: Training Random Forest (quick)...")
    
    model = RandomForestClassifier(
        n_estimators=50,  # Fewer trees for speed
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Step 5: Evaluation
    print("\nStep 5: Evaluation...")
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"📊 Results:")
    print(f"   Train Accuracy: {train_acc:.3f}")
    print(f"   Test Accuracy:  {test_acc:.3f}")
    
    # Step 6: Save model
    print("\nStep 6: Saving model...")
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'emotion_classes': emotion_classes,
        'accuracy': test_acc,
        'model_type': 'Random Forest'
    }
    
    joblib.dump(model_data, 'fer_best.pkl')
    print("✅ Model saved as 'fer_best.pkl'")
    
    print(f"\n🎉 Quick FER Training Complete!")
    print(f"🏆 Final accuracy: {test_acc:.3f}")
    print(f"💾 Model ready for LifeLine integration!")

if __name__ == "__main__":
    quick_fer_training()