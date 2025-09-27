#!/usr/bin/env python3
"""
Lightweight Scikit-Learn FER-2013 Training Script
=================================================
Trains a facial expression recognition model using scikit-learn
on the FER-2013 dataset for emotion detection in patient monitoring.
"""

import os
import zipfile
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import glob
from pathlib import Path

def extract_fer_dataset():
    """Extract fer.zip to fer_dataset folder if not already extracted."""
    dataset_path = "datasets/fer_dataset"
    zip_path = "datasets/fer.zip"
    
    if not os.path.exists(dataset_path):
        if not os.path.exists(zip_path):
            print(f"❌ Error: {zip_path} not found!")
            return False
            
        print(f"📦 Extracting {zip_path}...")
        os.makedirs(dataset_path, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        
        print(f"✅ Dataset extracted to {dataset_path}")
    else:
        print(f"✅ Dataset already exists at {dataset_path}")
    
    return True

def load_images_from_folder(folder_path, target_size=(48, 48)):
    """Load images from folder and return as flattened arrays with labels."""
    images = []
    labels = []
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return np.array([]), np.array([])
    
    # Get emotion classes (subfolder names)
    emotion_classes = sorted([d for d in os.listdir(folder_path) 
                             if os.path.isdir(os.path.join(folder_path, d))])
    
    print(f"📁 Found emotion classes: {emotion_classes}")
    
    class_to_idx = {cls: idx for idx, cls in enumerate(emotion_classes)}
    
    for emotion_class in emotion_classes:
        class_folder = os.path.join(folder_path, emotion_class)
        image_files = glob.glob(os.path.join(class_folder, "*.jpg")) + \
                     glob.glob(os.path.join(class_folder, "*.png")) + \
                     glob.glob(os.path.join(class_folder, "*.jpeg"))
        
        print(f"📸 Loading {len(image_files)} images from {emotion_class}...")
        
        for img_path in image_files:
            try:
                # Load and preprocess image
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to target size
                img = img.resize(target_size)
                
                # Convert to grayscale for simpler feature extraction
                img = img.convert('L')
                
                # Convert to numpy array and flatten
                img_array = np.array(img).flatten()
                
                # Normalize to [0, 1]
                img_array = img_array / 255.0
                
                images.append(img_array)
                labels.append(class_to_idx[emotion_class])
                
            except Exception as e:
                print(f"⚠️  Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels), emotion_classes

def train_fer_model():
    """Train facial expression recognition model using scikit-learn."""
    print("🚀 Starting FER-2013 Training with Scikit-Learn")
    print("=" * 50)
    
    # Step 1: Extract dataset
    if not extract_fer_dataset():
        return
    
    # Step 2: Load training and test data
    print("\n📊 Loading datasets...")
    
    # Try different possible folder structures
    possible_paths = [
        "datasets/fer_dataset/train",
        "datasets/fer_dataset/fer2013/train",
        "datasets/fer_dataset/Training",
        "datasets/fer_dataset/training"
    ]
    
    train_path = None
    for path in possible_paths:
        if os.path.exists(path):
            train_path = path
            break
    
    if train_path is None:
        print("❌ Could not find training folder. Checking dataset structure...")
        # List contents to help debug
        fer_path = "datasets/fer_dataset"
        if os.path.exists(fer_path):
            contents = os.listdir(fer_path)
            print(f"Contents of {fer_path}: {contents}")
            
            # Look one level deeper
            for item in contents:
                item_path = os.path.join(fer_path, item)
                if os.path.isdir(item_path):
                    sub_contents = os.listdir(item_path)
                    print(f"Contents of {item_path}: {sub_contents}")
        return
    
    # Find corresponding test path
    test_path = train_path.replace("train", "test").replace("Training", "Testing")
    
    print(f"📁 Training path: {train_path}")
    print(f"📁 Testing path: {test_path}")
    
    # Load training data
    X_train, y_train, emotion_classes = load_images_from_folder(train_path)
    
    if len(X_train) == 0:
        print("❌ No training data found!")
        return
    
    # Load test data
    X_test, y_test, _ = load_images_from_folder(test_path)
    
    if len(X_test) == 0:
        print("⚠️  No test data found, using train/validation split")
        # Split training data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
    
    print(f"✅ Loaded {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"🎭 Emotion classes: {emotion_classes}")
    print(f"📏 Feature dimension: {X_train.shape[1]}")
    
    # Step 3: Preprocessing
    print("\n🔧 Preprocessing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 4: Train multiple models and compare
    print("\n🎯 Training models...")
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            n_jobs=-1
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', 
            C=1.0, 
            gamma='scale', 
            random_state=42
        )
    }
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\n🏋️  Training {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"📊 {name} Results:")
        print(f"   Train Accuracy: {train_acc:.3f}")
        print(f"   Test Accuracy:  {test_acc:.3f}")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = model
            best_name = name
    
    # Step 5: Save best model
    print(f"\n💾 Saving best model ({best_name}, {best_accuracy:.3f} accuracy)...")
    
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'emotion_classes': emotion_classes,
        'accuracy': best_accuracy,
        'model_type': best_name
    }
    
    joblib.dump(model_data, 'fer_best.pkl')
    print("✅ Model saved as 'fer_best.pkl'")
    
    # Step 6: Detailed evaluation of best model
    print(f"\n📈 Detailed evaluation of {best_name}:")
    test_pred = best_model.predict(X_test_scaled)
    
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=emotion_classes))
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best model: {best_name}")
    print(f"🎯 Final accuracy: {best_accuracy:.3f}")
    print(f"💾 Model saved: fer_best.pkl")

def test_model():
    """Quick test of the saved model."""
    if os.path.exists('fer_best.pkl'):
        print("\n🧪 Testing saved model...")
        model_data = joblib.load('fer_best.pkl')
        print(f"✅ Loaded {model_data['model_type']} model")
        print(f"🎭 Emotion classes: {model_data['emotion_classes']}")
        print(f"🎯 Training accuracy: {model_data['accuracy']:.3f}")

if __name__ == "__main__":
    # Train the model
    train_fer_model()
    
    # Test loading the saved model
    test_model()
    
    print("\n" + "="*50)
    print("🎉 FER Training Complete!")
    print("Ready for integration with LifeLine patient monitoring system.")