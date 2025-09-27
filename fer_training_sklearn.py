#!/usr/bin/env python3
"""
Lightweight scikit-learn script to train on FER-2013 dataset for facial expression recognition.
Automatically handles dataset extraction, training, and model saving.
"""

import os
import zipfile
import numpy as np
from PIL import Image
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import time
from glob import glob

# Configuration
DATASET_ZIP = "datasets/fer.zip"
DATASET_DIR = "fer_dataset"
MODEL_SAVE_PATH = "fer_best.pkl"
SCALER_SAVE_PATH = "fer_scaler.pkl"
IMAGE_SIZE = (48, 48)
TEST_SIZE = 0.2
RANDOM_STATE = 42

def extract_dataset():
    """Extract the FER dataset if not already extracted."""
    if not os.path.exists(DATASET_DIR):
        print(f"📦 Extracting {DATASET_ZIP}...")
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)
        print(f"✅ Dataset extracted to {DATASET_DIR}")
    else:
        print(f"✅ Dataset already exists at {DATASET_DIR}")

def load_image_features(image_path):
    """Load and preprocess image, extract features."""
    try:
        # Load image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Try with PIL as backup
            img = Image.open(image_path).convert('L')
            img = np.array(img)
        
        # Resize to 48x48
        img = cv2.resize(img, IMAGE_SIZE)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Extract features:
        # 1. Raw pixel values (flattened)
        pixel_features = img.flatten()
        
        # 2. Histogram features
        hist = cv2.calcHist([img], [0], None, [16], [0, 1])
        hist_features = hist.flatten()
        
        # 3. Basic statistical features
        stats = [
            np.mean(img),
            np.std(img),
            np.min(img),
            np.max(img),
            np.median(img)
        ]
        
        # Combine all features
        features = np.concatenate([pixel_features, hist_features, stats])
        return features
        
    except Exception as e:
        print(f"❌ Error loading {image_path}: {e}")
        return None

def load_dataset():
    """Load the FER dataset and extract features."""
    print("📂 Loading FER dataset...")
    
    # Find all emotion classes
    train_dir = os.path.join(DATASET_DIR, "train")
    test_dir = os.path.join(DATASET_DIR, "test")
    
    # If direct train/test don't exist, look for the actual structure
    if not os.path.exists(train_dir):
        # Look for the actual extracted structure
        extracted_dirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
        if extracted_dirs:
            # Try to find train folder in subdirectories
            for subdir in extracted_dirs:
                potential_train = os.path.join(DATASET_DIR, subdir, "train")
                if os.path.exists(potential_train):
                    train_dir = potential_train
                    test_dir = os.path.join(DATASET_DIR, subdir, "test")
                    break
    
    X_all = []
    y_all = []
    class_names = []
    
    # Process training data
    if os.path.exists(train_dir):
        emotion_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        class_names = sorted(emotion_dirs)
        
        print(f"📋 Found emotion classes: {class_names}")
        
        for class_idx, emotion in enumerate(class_names):
            emotion_path = os.path.join(train_dir, emotion)
            image_files = glob(os.path.join(emotion_path, "*.jpg")) + \
                         glob(os.path.join(emotion_path, "*.png")) + \
                         glob(os.path.join(emotion_path, "*.jpeg"))
            
            print(f"📸 Processing {emotion}: {len(image_files)} images")
            
            for img_path in image_files:
                features = load_image_features(img_path)
                if features is not None:
                    X_all.append(features)
                    y_all.append(class_idx)
    
    # Process test data if available
    if os.path.exists(test_dir):
        print("📂 Processing test set...")
        for class_idx, emotion in enumerate(class_names):
            emotion_path = os.path.join(test_dir, emotion)
            if os.path.exists(emotion_path):
                image_files = glob(os.path.join(emotion_path, "*.jpg")) + \
                             glob(os.path.join(emotion_path, "*.png")) + \
                             glob(os.path.join(emotion_path, "*.jpeg"))
                
                for img_path in image_files:
                    features = load_image_features(img_path)
                    if features is not None:
                        X_all.append(features)
                        y_all.append(class_idx)
    
    if not X_all:
        print("❌ No images found! Check dataset structure.")
        return None, None, None, None
    
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    print(f"✅ Loaded {len(X_all)} samples with {X_all.shape[1]} features each")
    print(f"📊 Classes: {len(class_names)} emotions")
    
    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
    )
    
    return X_train, X_val, y_train, y_val, class_names

def train_model(X_train, X_val, y_train, y_val, class_names):
    """Train the facial expression recognition model."""
    print("🧠 Training Random Forest model...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create and train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    # Evaluate
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"⏱️  Training completed in {training_time:.2f} seconds")
    print(f"📈 Training Accuracy: {train_acc:.4f}")
    print(f"📊 Validation Accuracy: {val_acc:.4f}")
    
    # Detailed classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_val, val_pred, target_names=class_names))
    
    # Save model and scaler
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"💾 Model saved to {MODEL_SAVE_PATH}")
    print(f"💾 Scaler saved to {SCALER_SAVE_PATH}")
    
    return model, scaler, val_acc

def main():
    """Main training pipeline."""
    print("🚀 Starting FER-2013 Facial Expression Recognition Training")
    print("=" * 60)
    
    # Extract dataset
    extract_dataset()
    
    # Load and preprocess data
    X_train, X_val, y_train, y_val, class_names = load_dataset()
    
    if X_train is None:
        print("❌ Failed to load dataset. Please check the dataset structure.")
        return
    
    # Train model
    model, scaler, best_acc = train_model(X_train, X_val, y_train, y_val, class_names)
    
    print("\n🎉 Training Complete!")
    print(f"🏆 Best Validation Accuracy: {best_acc:.4f}")
    print(f"📁 Model files: {MODEL_SAVE_PATH}, {SCALER_SAVE_PATH}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print(f"🔍 Top feature types by importance:")
        feature_names = ['pixels', 'histogram', 'statistics']
        # Simplified importance display
        pixel_importance = np.mean(model.feature_importances_[:IMAGE_SIZE[0]*IMAGE_SIZE[1]])
        hist_importance = np.mean(model.feature_importances_[IMAGE_SIZE[0]*IMAGE_SIZE[1]:IMAGE_SIZE[0]*IMAGE_SIZE[1]+16])
        stats_importance = np.mean(model.feature_importances_[-5:])
        
        print(f"   Pixels: {pixel_importance:.4f}")
        print(f"   Histogram: {hist_importance:.4f}")
        print(f"   Statistics: {stats_importance:.4f}")

if __name__ == "__main__":
    main()