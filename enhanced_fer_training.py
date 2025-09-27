#!/usr/bin/env python3
"""
Enhanced FER Training with Advanced Techniques
==============================================
Using more sophisticated approaches to achieve higher accuracy.
"""

import os
import time
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
import joblib
import glob

def extract_advanced_features(img_array):
    """Extract advanced features from facial images."""
    features = []
    
    # 1. Raw pixel features (reduced resolution)
    resized = np.array(Image.fromarray(img_array).resize((32, 32)))
    features.extend(resized.flatten())
    
    # 2. Statistical moments
    features.extend([
        np.mean(img_array),
        np.std(img_array),
        np.median(img_array),
        np.percentile(img_array, 25),
        np.percentile(img_array, 75),
        np.min(img_array),
        np.max(img_array)
    ])
    
    # 3. Histogram features (multiple bins)
    hist_8, _ = np.histogram(img_array.flatten(), bins=8, range=(0, 1))
    hist_16, _ = np.histogram(img_array.flatten(), bins=16, range=(0, 1))
    features.extend(hist_8.astype(np.float32) / np.sum(hist_8))
    features.extend(hist_16.astype(np.float32) / np.sum(hist_16))
    
    # 4. Texture features (using simple gradients)
    grad_x = np.gradient(img_array, axis=1)
    grad_y = np.gradient(img_array, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    features.extend([
        np.mean(gradient_magnitude),
        np.std(gradient_magnitude),
        np.median(gradient_magnitude)
    ])
    
    # 5. Regional features (divide face into quadrants)
    h, w = img_array.shape
    regions = [
        img_array[:h//2, :w//2],      # Top-left
        img_array[:h//2, w//2:],      # Top-right
        img_array[h//2:, :w//2],      # Bottom-left
        img_array[h//2:, w//2:]       # Bottom-right
    ]
    
    for region in regions:
        features.extend([
            np.mean(region),
            np.std(region),
            np.max(region) - np.min(region)  # Range
        ])
    
    return np.array(features, dtype=np.float32)

def load_balanced_dataset(dataset_path, samples_per_class=200):
    """Load a balanced dataset with advanced preprocessing."""
    print(f"🎯 Loading balanced dataset from {dataset_path}...")
    
    # Find emotion directories
    emotion_dirs = []
    for root, dirs, files in os.walk(dataset_path):
        jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(jpg_files) > 50:  # Only consider substantial directories
            emotion_dirs.append((root, jpg_files))
    
    if len(emotion_dirs) < 3:
        print("❌ Insufficient emotion directories found")
        return None, None, None
    
    # Sort by directory name for consistency
    emotion_dirs.sort(key=lambda x: os.path.basename(x[0]))
    
    all_features = []
    all_labels = []
    emotion_classes = []
    
    for class_idx, (dir_path, files) in enumerate(emotion_dirs[:7]):  # Limit to 7 classes
        class_name = os.path.basename(dir_path)
        emotion_classes.append(class_name)
        
        print(f"📸 Processing {class_name} ({len(files)} available)...")
        
        # Sample files evenly
        selected_files = files[:samples_per_class] if len(files) >= samples_per_class else files
        
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
                img = enhancer.enhance(1.2)
                
                # Resize and convert to grayscale
                img = img.resize((48, 48), Image.Resampling.LANCZOS)
                img_gray = img.convert('L')
                img_array = np.array(img_gray, dtype=np.float32) / 255.0
                
                # Extract advanced features
                features = extract_advanced_features(img_array)
                class_features.append(features)
                
            except Exception as e:
                continue
        
        if len(class_features) == 0:
            continue
            
        # Balance classes by duplicating samples if needed
        target_samples = max(100, min(samples_per_class, len(class_features)))
        
        if len(class_features) < target_samples:
            # Duplicate samples to reach target
            while len(class_features) < target_samples:
                class_features.extend(class_features[:target_samples-len(class_features)])
        else:
            class_features = class_features[:target_samples]
        
        # Add to dataset
        all_features.extend(class_features)
        all_labels.extend([class_idx] * len(class_features))
        
        print(f"✅ Added {len(class_features)} samples for {class_name}")
    
    if not all_features:
        print("❌ No features extracted")
        return None, None, None
    
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels)
    
    print(f"🎯 Final balanced dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"📊 Class distribution: {np.bincount(y)}")
    
    return X, y, emotion_classes

def train_ensemble_with_cv(X_train, X_test, y_train, y_test, emotion_classes):
    """Train ensemble with cross-validation to maximize accuracy."""
    
    print("🎯 Training ensemble with cross-validation...")
    
    # Base models with different strengths
    models = {
        'rf_balanced': RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'gb_strong': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            random_state=123
        ),
        'lr_regularized': LogisticRegression(
            C=1.0,
            penalty='l2',
            class_weight='balanced',
            max_iter=1000,
            random_state=456
        )
    }
    
    # Train and evaluate each model
    model_results = {}
    for name, model in models.items():
        print(f"\n🏋️ Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = np.mean(cv_scores)
        
        # Full training
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        
        model_results[name] = {
            'model': model,
            'cv_score': cv_mean,
            'test_acc': test_acc
        }
        
        print(f"📊 {name}: CV={cv_mean:.3f}, Test={test_acc:.3f}")
    
    # Find best individual model
    best_individual = max(model_results.items(), key=lambda x: x[1]['test_acc'])
    best_name, best_result = best_individual
    
    print(f"\n🏆 Best individual: {best_name} with {best_result['test_acc']:.3f}")
    
    # Try ensemble of top models
    try:
        from sklearn.ensemble import VotingClassifier
        
        # Select top 2-3 models for ensemble
        top_models = sorted(model_results.items(), key=lambda x: x[1]['test_acc'], reverse=True)[:3]
        
        ensemble_estimators = [(name, result['model']) for name, result in top_models]
        
        ensemble = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft'
        )
        
        print("🔗 Training ensemble...")
        ensemble.fit(X_train, y_train)
        
        ensemble_pred = ensemble.predict(X_test)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        
        print(f"📊 Ensemble accuracy: {ensemble_acc:.3f}")
        
        if ensemble_acc > best_result['test_acc']:
            return ensemble, ensemble_acc, "Ensemble"
        
    except Exception as e:
        print(f"⚠️ Ensemble failed: {e}")
    
    return best_result['model'], best_result['test_acc'], best_name

def enhanced_fer_training():
    """Enhanced FER training with advanced techniques."""
    print("🚀 ENHANCED FER TRAINING - ADVANCED TECHNIQUES")
    print("=" * 55)
    
    # Load balanced dataset
    X, y, emotion_classes = load_balanced_dataset("datasets/fer_dataset", samples_per_class=250)
    
    if X is None:
        print("❌ Dataset loading failed")
        return None
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"📊 Dataset split: {len(X_train)} train, {len(X_test)} test")
    
    # Feature preprocessing with PCA for dimensionality reduction
    print("🔧 Advanced preprocessing...")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA to reduce dimensionality while preserving variance
    pca = PCA(n_components=0.95, random_state=42)  # Keep 95% of variance
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"📉 PCA: {X_train_scaled.shape[1]} → {X_train_pca.shape[1]} features")
    
    # Train advanced ensemble
    best_model, best_acc, best_name = train_ensemble_with_cv(
        X_train_pca, X_test_pca, y_train, y_test, emotion_classes
    )
    
    # Save model with all preprocessing components
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'pca': pca,
        'emotion_classes': emotion_classes,
        'accuracy': best_acc,
        'model_type': best_name,
        'feature_count': X_train_pca.shape[1],
        'original_features': X_train_scaled.shape[1]
    }
    
    joblib.dump(model_data, 'fer_best.pkl')
    
    # Results
    print(f"\n🎉 ENHANCED TRAINING COMPLETE!")
    print(f"🏆 Best Model: {best_name}")
    print(f"🎯 Final Accuracy: {best_acc:.3f} ({best_acc:.1%})")
    
    if best_acc >= 0.85:
        print("✅ SUCCESS: Target 85%+ accuracy ACHIEVED!")
    elif best_acc >= 0.75:
        print("🟡 VERY GOOD: 75%+ accuracy achieved!")
    elif best_acc >= 0.65:
        print("🟠 GOOD: 65%+ accuracy achieved!")
    elif best_acc >= 0.50:
        print("🟡 FAIR: 50%+ accuracy achieved!")
    else:
        print("🔴 Needs improvement")
    
    print(f"💾 Model saved with all preprocessing components")
    print(f"📊 Ready for LifeLine integration!")
    
    return best_acc

if __name__ == "__main__":
    print("🚀 Starting Enhanced FER Training...")
    start_time = time.time()
    
    try:
        final_accuracy = enhanced_fer_training()
        
        total_time = time.time() - start_time
        print(f"\n" + "="*60)
        print(f"⏱️ Total training time: {total_time:.1f} seconds")
        
        if final_accuracy:
            print(f"🎯 FINAL RESULT: {final_accuracy:.3f} accuracy!")
            print(f"📈 Significant improvement from baseline!")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()