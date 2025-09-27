# train_fall_model.py - Lightweight Fall Detection Training Pipeline
import os
import glob
import json
import cv2
import numpy as np
from datetime import datetime

def load_dataset():
    """Load the fall detection dataset from archive"""
    print("📁 Loading fall detection dataset...")
    
    # Path to dataset
    dataset_path = "../datasets/archive/fall_dataset/images/train"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    # Load images and labels
    fall_images = []
    normal_images = []
    
    for img_file in glob.glob(os.path.join(dataset_path, "*.jpg")):
        filename = os.path.basename(img_file).lower()
        
        # Load image
        img = cv2.imread(img_file)
        if img is None:
            continue
            
        # Classify based on filename
        if 'fall' in filename and 'not' not in filename:
            fall_images.append({
                'path': img_file,
                'image': img,
                'label': 'fall'
            })
        elif 'not fallen' in filename or 'normal' in filename:
            normal_images.append({
                'path': img_file,
                'image': img,
                'label': 'normal'
            })
    
    print(f"✅ Loaded {len(fall_images)} fall images, {len(normal_images)} normal images")
    return fall_images, normal_images

def extract_features(image):
    """Extract simple features for fall detection"""
    h, w, _ = image.shape
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Basic features
    features = {
        'aspect_ratio': w / h,
        'brightness': np.mean(gray),
        'contrast': np.std(gray),
        'bottom_heavy': np.mean(gray[h//2:, :]) / np.mean(gray[:h//2, :]),  # Bottom vs top brightness
        'horizontal_edges': len(cv2.findContours(cv2.Canny(gray, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    }
    
    return features

def train_simple_model(fall_images, normal_images):
    """Train a simple rule-based model"""
    print("🧠 Training lightweight fall detection model...")
    
    # Extract features from training data
    fall_features = []
    normal_features = []
    
    # Process fall images
    for item in fall_images[:50]:  # Use first 50 for speed
        features = extract_features(item['image'])
        fall_features.append(features)
    
    # Process normal images  
    for item in normal_images[:50]:  # Use first 50 for speed
        features = extract_features(item['image'])
        normal_features.append(features)
    
    # Calculate thresholds
    model = {}
    
    for feature_name in fall_features[0].keys():
        fall_values = [f[feature_name] for f in fall_features]
        normal_values = [f[feature_name] for f in normal_features]
        
        fall_mean = np.mean(fall_values)
        normal_mean = np.mean(normal_values)
        threshold = (fall_mean + normal_mean) / 2
        
        model[feature_name] = {
            'threshold': float(threshold),
            'fall_mean': float(fall_mean),
            'normal_mean': float(normal_mean),
            'higher_indicates_fall': bool(fall_mean > normal_mean)
        }
    
    print("✅ Model training completed")
    return model

def test_model(model, fall_images, normal_images):
    """Test the trained model"""
    print("🧪 Testing model performance...")
    
    correct_predictions = 0
    total_predictions = 0
    
    # Test on fall images
    for item in fall_images[50:70]:  # Test on different set
        features = extract_features(item['image'])
        prediction = predict_fall(model, features)
        
        if prediction['is_fall']:
            correct_predictions += 1
        total_predictions += 1
    
    # Test on normal images
    for item in normal_images[50:70]:  # Test on different set
        features = extract_features(item['image'])
        prediction = predict_fall(model, features)
        
        if not prediction['is_fall']:
            correct_predictions += 1
        total_predictions += 1
    
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"📊 Model Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
    
    return accuracy

def predict_fall(model, features):
    """Predict if an image shows a fall"""
    fall_score = 0
    total_features = len(model)
    
    details = {}
    
    for feature_name, feature_model in model.items():
        feature_value = features[feature_name]
        threshold = feature_model['threshold']
        higher_indicates_fall = feature_model['higher_indicates_fall']
        
        if higher_indicates_fall:
            contributes_to_fall = feature_value > threshold
        else:
            contributes_to_fall = feature_value < threshold
        
        if contributes_to_fall:
            fall_score += 1
        
        details[feature_name] = {
            'value': feature_value,
            'threshold': threshold,
            'contributes_to_fall': contributes_to_fall
        }
    
    confidence = fall_score / total_features
    is_fall = confidence > 0.5  # Majority vote
    
    return {
        'is_fall': is_fall,
        'confidence': confidence,
        'fall_score': fall_score,
        'total_features': total_features,
        'details': details
    }

def save_model(model, filename='fall_model.json'):
    """Save the trained model"""
    model_data = {
        'model': model,
        'training_date': datetime.now().isoformat(),
        'model_type': 'simple_rule_based',
        'version': '1.0'
    }
    
    with open(filename, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"💾 Model saved as: {filename}")

def load_model(filename='fall_model.json'):
    """Load a trained model"""
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'r') as f:
        model_data = json.load(f)
    
    return model_data['model']

def main():
    """Main training pipeline"""
    print("🏥 Lightweight Fall Detection Training Pipeline")
    print("=" * 50)
    
    try:
        # Load dataset
        fall_images, normal_images = load_dataset()
        
        if len(fall_images) == 0 or len(normal_images) == 0:
            print("❌ Insufficient data for training")
            return False
        
        # Train model
        model = train_simple_model(fall_images, normal_images)
        
        # Test model
        accuracy = test_model(model, fall_images, normal_images)
        
        # Save model
        save_model(model)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📈 Final accuracy: {accuracy:.1f}%")
        print(f"💾 Model saved as: fall_model.json")
        
        # Show model summary
        print(f"\n📋 Model Summary:")
        for feature, params in model.items():
            direction = "↑ higher" if params['higher_indicates_fall'] else "↓ lower"
            print(f"   {feature}: {direction} values indicate fall (threshold: {params['threshold']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n💡 Next steps:")
        print("1. Use the trained model in fall_detection.py")
        print("2. Test with: python test_fall_model.py")
        print("3. Integrate with the monitoring system")