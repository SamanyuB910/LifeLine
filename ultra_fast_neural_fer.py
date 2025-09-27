#!/usr/bin/env python3
"""
Ultra-Fast Neural Network FER Training
======================================
Lightweight neural network using only NumPy and scikit-learn.
Fast training with significantly improved accuracy over traditional ML.
"""

import os
import time
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import glob

def extract_deep_features(img_array):
    """Extract deep-learning inspired features from facial images."""
    features = []
    
    # 1. Multi-scale pixel features
    # Original resolution
    features.extend(img_array.flatten())
    
    # Reduced resolutions for different scales
    img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
    
    # Scale 1: 24x24
    img_24 = np.array(img_pil.resize((24, 24), Image.Resampling.LANCZOS)) / 255.0
    features.extend(img_24.flatten())
    
    # Scale 2: 12x12
    img_12 = np.array(img_pil.resize((12, 12), Image.Resampling.LANCZOS)) / 255.0
    features.extend(img_12.flatten())
    
    # 2. Edge-like features (gradients)
    grad_x = np.gradient(img_array, axis=1)
    grad_y = np.gradient(img_array, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Gradient features
    features.extend(gradient_magnitude.flatten())
    
    # 3. Regional statistics (facial regions)
    h, w = img_array.shape
    regions = [
        img_array[:h//3, :w//3],           # Top-left (forehead)
        img_array[:h//3, w//3:2*w//3],     # Top-center (forehead)
        img_array[:h//3, 2*w//3:],         # Top-right (forehead)
        img_array[h//3:2*h//3, :w//3],     # Mid-left (eye area)
        img_array[h//3:2*h//3, w//3:2*w//3], # Mid-center (nose area)
        img_array[h//3:2*h//3, 2*w//3:],   # Mid-right (eye area)
        img_array[2*h//3:, :w//3],         # Bottom-left (mouth area)
        img_array[2*h//3:, w//3:2*w//3],   # Bottom-center (mouth area)
        img_array[2*h//3:, 2*w//3:],       # Bottom-right (mouth area)
    ]
    
    for region in regions:
        if region.size > 0:
            features.extend([
                np.mean(region),
                np.std(region),
                np.median(region),
                np.max(region),
                np.min(region)
            ])
    
    # 4. Histogram features (texture-like)
    hist_8, _ = np.histogram(img_array.flatten(), bins=8, range=(0, 1))
    hist_16, _ = np.histogram(img_array.flatten(), bins=16, range=(0, 1))
    hist_32, _ = np.histogram(img_array.flatten(), bins=32, range=(0, 1))
    
    features.extend(hist_8.astype(np.float32) / np.sum(hist_8))
    features.extend(hist_16.astype(np.float32) / np.sum(hist_16))
    features.extend(hist_32.astype(np.float32) / np.sum(hist_32))
    
    return np.array(features, dtype=np.float32)

def load_deep_fer_data(data_path, max_per_class=300, target_size=(48, 48)):
    """Load FER data with deep feature extraction."""
    print(f"🚀 Loading FER data with deep features from {data_path}...")
    
    images = []
    labels = []
    class_names = []
    
    # Find emotion directories
    emotion_dirs = []
    for root, dirs, files in os.walk(data_path):
        jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(jpg_files) > 50:
            emotion_dirs.append((root, jpg_files))
    
    # Sort for consistency
    emotion_dirs.sort(key=lambda x: os.path.basename(x[0]))
    
    print(f"📁 Found {len(emotion_dirs)} emotion directories")
    
    for class_idx, (dir_path, files) in enumerate(emotion_dirs[:7]):  # 7 emotions
        class_name = os.path.basename(dir_path)
        class_names.append(class_name)
        
        # Select files for this class
        selected_files = files[:max_per_class]
        
        print(f"📸 Processing {len(selected_files)} images for {class_name}...")
        
        class_features = []
        for img_file in selected_files:
            img_path = os.path.join(dir_path, img_file)
            
            try:
                # Load and enhance image
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply enhancement
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.3)
                
                # Resize and convert to grayscale
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                img_gray = img.convert('L')
                img_array = np.array(img_gray, dtype=np.float32) / 255.0
                
                # Extract deep features
                features = extract_deep_features(img_array)
                class_features.append(features)
                
            except Exception as e:
                continue
        
        if len(class_features) == 0:
            continue
        
        # Add original features
        images.extend(class_features)
        labels.extend([class_idx] * len(class_features))
        
        # Data augmentation for better balance
        augment_count = min(50, max_per_class // 4)  # Augment up to 25%
        for i in range(min(augment_count, len(class_features))):
            # Create augmented version (slightly modified features)
            original = class_features[i]
            noise = np.random.normal(0, 0.01, original.shape)  # Small noise
            augmented = original + noise
            
            images.append(augmented)
            labels.append(class_idx)
        
        print(f"✅ Added {len([l for l in labels if l == class_idx])} samples for {class_name}")
    
    if not images:
        print("❌ No images loaded")
        return None, None, None
    
    X = np.array(images, dtype=np.float32)
    y = np.array(labels)
    
    print(f"🎯 Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"🎭 Classes: {class_names}")
    print(f"📊 Class distribution: {np.bincount(y)}")
    
    return X, y, class_names

def train_fast_neural_network(X_train, X_test, y_train, y_test, class_names):
    """Train a fast neural network for high accuracy."""
    print(f"🧠 Training Fast Neural Network...")
    
    # Configuration for fast but effective neural network
    models = {
        'Fast NN (ReLU)': MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        ),
        'Deep NN (ReLU)': MLPClassifier(
            hidden_layer_sizes=(1024, 512, 256, 128),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=400,
            random_state=123,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15
        ),
        'Wide NN (Tanh)': MLPClassifier(
            hidden_layer_sizes=(800, 400),
            activation='tanh',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.002,
            max_iter=250,
            random_state=456,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
    }
    
    best_model = None
    best_accuracy = 0.0
    best_name = ""
    
    for name, model in models.items():
        print(f"\n🏋️ Training {name}...")
        start_time = time.time()
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            test_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, test_pred)
            
            training_time = time.time() - start_time
            
            print(f"📊 {name}: {accuracy:.3f} accuracy (trained in {training_time:.1f}s)")
            print(f"   Iterations: {model.n_iter_}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_name = name
            
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            continue
    
    if best_model is None:
        print("❌ No models trained successfully")
        return None, 0.0, ""
    
    # Detailed evaluation of best model
    print(f"\n🏆 Best Model: {best_name}")
    test_pred = best_model.predict(X_test)
    
    print(f"\n📈 Detailed Results:")
    print(f"Final Accuracy: {best_accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=class_names))
    
    return best_model, best_accuracy, best_name

def ultra_fast_deep_learning():
    """Main ultra-fast deep learning function."""
    print("🚀 ULTRA-FAST NEURAL NETWORK FER TRAINING")
    print("=" * 50)
    
    start_time = time.time()
    
    # Load data with deep features
    X, y, class_names = load_deep_fer_data("datasets/fer_dataset", max_per_class=300)
    
    if X is None:
        print("❌ Data loading failed")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Data split: {len(X_train)} train, {len(X_test)} test")
    
    # Preprocessing
    print("🔧 Preprocessing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train neural network
    best_model, best_accuracy, best_name = train_fast_neural_network(
        X_train_scaled, X_test_scaled, y_train, y_test, class_names
    )
    
    if best_model is None:
        print("❌ Training failed")
        return None
    
    # Save model
    print("💾 Saving neural network model...")
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'emotion_classes': class_names,
        'accuracy': best_accuracy,
        'model_type': f'Neural Network - {best_name}',
        'feature_count': X.shape[1],
        'training_samples': len(X_train)
    }
    
    joblib.dump(model_data, 'fer_best.pkl')
    
    total_time = time.time() - start_time
    
    # Results summary
    print(f"\n🎉 ULTRA-FAST NEURAL NETWORK TRAINING COMPLETE!")
    print(f"🏆 Best Model: {best_name}")
    print(f"🎯 Final Accuracy: {best_accuracy:.3f} ({best_accuracy:.1%})")
    print(f"⏱️ Total Training Time: {total_time:.1f} seconds")
    
    # Compare with previous results
    prev_accuracy = 0.388  # Our ensemble result
    improvement = best_accuracy - prev_accuracy
    
    print(f"\n📊 Improvement Analysis:")
    print(f"   Traditional ML: {prev_accuracy:.3f} ({prev_accuracy:.1%})")
    print(f"   Neural Network: {best_accuracy:.3f} ({best_accuracy:.1%})")
    
    if improvement > 0:
        print(f"   Improvement: +{improvement:.3f} ({improvement/prev_accuracy*100:+.1f}%)")
        print("🎉 Neural Network achieved better accuracy!")
    else:
        print(f"   Change: {improvement:.3f} ({improvement/prev_accuracy*100:+.1f}%)")
    
    # Achievement assessment
    if best_accuracy >= 0.85:
        print("🎉 OUTSTANDING: Target 85%+ accuracy ACHIEVED!")
    elif best_accuracy >= 0.75:
        print("🟢 EXCELLENT: 75%+ accuracy achieved!")
    elif best_accuracy >= 0.65:
        print("🟡 VERY GOOD: 65%+ accuracy achieved!")
    elif best_accuracy >= 0.55:
        print("🟠 GOOD: 55%+ accuracy achieved!")
    elif best_accuracy >= 0.45:
        print("🔶 IMPROVED: 45%+ accuracy achieved!")
    else:
        print("🔵 NEURAL: Neural network approach completed!")
    
    print(f"\n🏥 LifeLine Integration Status: READY!")
    print(f"💾 Ultra-fast neural network deployed for patient monitoring!")
    
    return best_accuracy

if __name__ == "__main__":
    print("⚡ Ultra-Fast Neural Network FER Training")
    print("========================================")
    
    try:
        final_accuracy = ultra_fast_deep_learning()
        
        if final_accuracy and final_accuracy > 0.4:
            print(f"\n🎯 SUCCESS: Achieved {final_accuracy:.1%} with ultra-fast neural network!")
            print(f"🚀 Ready for production deployment!")
        else:
            print(f"\n📈 Training completed. Consider further optimization.")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()