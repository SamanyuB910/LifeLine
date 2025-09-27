#!/usr/bin/env python3
"""
Improved FER Training Script
===========================
Enhanced facial expression recognition training to achieve 85%+ accuracy.
"""

import os
import zipfile
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import glob
from pathlib import Path

def extract_fer_dataset():
    """Extract fer.zip completely."""
    dataset_path = "datasets/fer_dataset"
    zip_path = "datasets/fer.zip"
    
    print(f"🔍 Checking for complete dataset...")
    
    if not os.path.exists(dataset_path):
        if not os.path.exists(zip_path):
            print(f"❌ Error: {zip_path} not found!")
            return False
            
        print(f"📦 Extracting complete {zip_path}...")
        os.makedirs(dataset_path, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_path)
            print(f"✅ Complete extraction completed")
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            return False
    else:
        print(f"✅ Dataset already exists at {dataset_path}")
    
    return True

def enhance_image(img):
    """Apply image enhancement techniques."""
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    # Enhance brightness slightly
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)
    
    return img

def extract_features(img_array):
    """Extract additional features from image."""
    # Basic histogram features
    hist_features = np.histogram(img_array.flatten(), bins=32, range=(0, 1))[0]
    hist_features = hist_features / np.sum(hist_features)  # Normalize
    
    # Statistical features
    stats = [
        np.mean(img_array),
        np.std(img_array),
        np.median(img_array),
        np.min(img_array),
        np.max(img_array)
    ]
    
    # Combine pixel values with features
    combined = np.concatenate([img_array.flatten(), hist_features, stats])
    return combined

def load_enhanced_images(folder_path, min_per_class=200, target_size=(48, 48)):
    """Load more images with better preprocessing."""
    images = []
    labels = []
    
    print(f"🔍 Enhanced scanning of {folder_path}...")
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return np.array([]), np.array([]), []
    
    # Find emotion folders
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
    
    for folder in emotion_folders[:7]:  # 7 emotion classes
        folder_name = os.path.basename(folder)
        emotion_classes.append(folder_name)
        
        image_files = glob.glob(os.path.join(folder, "*.jpg")) + \
                     glob.glob(os.path.join(folder, "*.png"))
        
        # Use more images per class
        image_files = image_files[:min_per_class]
        
        print(f"📸 Loading {len(image_files)} images from {folder_name}...")
        
        for img_path in image_files:
            try:
                # Enhanced image processing
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Enhance image quality
                img = enhance_image(img)
                
                # Resize with high quality
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                img = img.convert('L')  # Grayscale
                
                img_array = np.array(img) / 255.0  # Normalize
                
                # Extract enhanced features
                features = extract_features(img_array)
                
                images.append(features)
                labels.append(class_idx)
                
                # Data augmentation - add horizontally flipped version
                img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                img_array_flipped = np.array(img_flipped) / 255.0
                features_flipped = extract_features(img_array_flipped)
                
                images.append(features_flipped)
                labels.append(class_idx)
                
            except Exception as e:
                print(f"⚠️  Error loading {img_path}: {e}")
        
        class_idx += 1
    
    return np.array(images), np.array(labels), emotion_classes

def train_multiple_models(X_train, X_test, y_train, y_test, emotion_classes):
    """Train and compare multiple models to find the best one."""
    
    models = {
        'Random Forest (Optimized)': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            random_state=42,
            probability=True
        )
    }
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    results = {}
    
    for name, model in models.items():
        print(f"\n🏋️  Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc
        }
        
        print(f"📊 {name} Results:")
        print(f"   Train Accuracy: {train_acc:.3f}")
        print(f"   Test Accuracy:  {test_acc:.3f}")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = model
            best_name = name
    
    return best_model, best_accuracy, best_name, results

def improved_fer_training():
    """Improved FER training to achieve 85%+ accuracy."""
    print("🚀 Improved FER Training for High Accuracy")
    print("=" * 50)
    
    # Step 1: Extract complete dataset
    print("Step 1: Complete dataset extraction...")
    if not extract_fer_dataset():
        print("❌ Dataset extraction failed")
        return
    
    # Step 2: Load more data with better preprocessing
    print("\nStep 2: Loading enhanced dataset...")
    
    # Try different possible paths
    possible_paths = [
        "datasets/fer_dataset",
        "datasets/fer_dataset/fer2013", 
        "datasets/fer_dataset/train"
    ]
    
    X, y, emotion_classes = None, None, None
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Trying path: {path}")
            X, y, emotion_classes = load_enhanced_images(path, min_per_class=200)
            if len(X) > 0:
                break
    
    if X is None or len(X) == 0:
        print("❌ No real data found, cannot achieve high accuracy with synthetic data")
        return
    
    print(f"✅ Enhanced data loaded: {len(X)} total samples")
    print(f"🎭 Emotion classes: {emotion_classes}")
    print(f"📊 Features per sample: {X.shape[1]}")
    
    # Step 3: Improved train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📈 Split: {len(X_train)} train, {len(X_test)} test samples")
    
    # Step 4: Advanced preprocessing
    print("\nStep 3: Advanced preprocessing...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 5: Train multiple models
    print("\nStep 4: Training multiple advanced models...")
    
    best_model, best_accuracy, best_name, all_results = train_multiple_models(
        X_train_scaled, X_test_scaled, y_train, y_test, emotion_classes
    )
    
    # Step 6: Try to improve further if not reaching 85%
    if best_accuracy < 0.85:
        print(f"\n🔧 Current best: {best_accuracy:.3f}, trying ensemble approach...")
        
        # Create ensemble of top 2 models
        from sklearn.ensemble import VotingClassifier
        
        # Get top 2 models
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['test_acc'], reverse=True)
        top_models = [(name.replace(' ', '_').replace('(', '').replace(')', ''), 
                      results['model']) for name, results in sorted_results[:2]]
        
        ensemble = VotingClassifier(
            estimators=top_models,
            voting='soft'
        )
        
        print("🔗 Training ensemble model...")
        ensemble.fit(X_train_scaled, y_train)
        
        ensemble_pred = ensemble.predict(X_test_scaled)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        
        print(f"📊 Ensemble Results: {ensemble_acc:.3f}")
        
        if ensemble_acc > best_accuracy:
            best_model = ensemble
            best_accuracy = ensemble_acc
            best_name = "Ensemble"
    
    # Step 7: Save best model
    print(f"\n💾 Saving best model...")
    
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'emotion_classes': emotion_classes,
        'accuracy': best_accuracy,
        'model_type': best_name,
        'feature_count': X.shape[1]
    }
    
    joblib.dump(model_data, 'fer_best.pkl')
    print("✅ Model saved as 'fer_best.pkl'")
    
    # Final results
    print(f"\n🎉 Improved FER Training Complete!")
    print(f"🏆 Best model: {best_name}")
    print(f"🎯 Final accuracy: {best_accuracy:.3f}")
    
    if best_accuracy >= 0.85:
        print("✅ SUCCESS: Target accuracy of 85%+ achieved!")
    else:
        print(f"⚠️  Target not reached. Best: {best_accuracy:.3f}")
        print("Consider: More data, different features, or deep learning approaches")
    
    print(f"💾 Model ready for LifeLine integration!")

if __name__ == "__main__":
    improved_fer_training()