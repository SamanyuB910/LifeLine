# test_fall_model.py - Test the trained fall detection model
import json
import cv2
import numpy as np
from train_fall_model import extract_features, predict_fall, load_model

def test_trained_model():
    """Test the trained fall detection model"""
    print("🧪 Testing Trained Fall Detection Model")
    print("=" * 40)
    
    # Load the trained model
    model = load_model('fall_model.json')
    if model is None:
        print("❌ No trained model found. Run train_fall_model.py first.")
        return False
    
    print("✅ Model loaded successfully")
    
    # Create test scenarios
    test_scenarios = [
        {
            'name': 'Normal Standing Person',
            'image': create_test_image(mode='standing'),
            'expected': False
        },
        {
            'name': 'Fallen Person (Horizontal)',
            'image': create_test_image(mode='fallen'),
            'expected': True
        },
        {
            'name': 'Leaning Person',
            'image': create_test_image(mode='leaning'),
            'expected': True
        }
    ]
    
    print(f"\n🎯 Testing {len(test_scenarios)} scenarios:")
    print("-" * 40)
    
    correct_predictions = 0
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        
        # Extract features
        features = extract_features(scenario['image'])
        
        # Make prediction
        prediction = predict_fall(model, features)
        
        # Check accuracy
        is_correct = prediction['is_fall'] == scenario['expected']
        if is_correct:
            correct_predictions += 1
        
        # Display results
        print(f"   Expected: {'Fall' if scenario['expected'] else 'Normal'}")
        print(f"   Predicted: {'Fall' if prediction['is_fall'] else 'Normal'}")
        print(f"   Confidence: {prediction['confidence']:.3f}")
        print(f"   Result: {'✅ Correct' if is_correct else '❌ Incorrect'}")
        
        # Show key features
        key_features = sorted(prediction['details'].items(), 
                            key=lambda x: abs(x[1]['value'] - x[1]['threshold']), 
                            reverse=True)[:3]
        
        print("   Key features:")
        for feature_name, feature_data in key_features:
            status = "📈 High" if feature_data['contributes_to_fall'] else "📉 Low"
            print(f"     {feature_name}: {feature_data['value']:.3f} {status}")
    
    accuracy = (correct_predictions / len(test_scenarios)) * 100
    print(f"\n📊 Test Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(test_scenarios)})")
    
    return accuracy > 50  # Pass if > 50% accurate

def create_test_image(mode='standing'):
    """Create synthetic test images for different scenarios"""
    if mode == 'standing':
        # Vertical person (taller than wide)
        img = np.random.randint(80, 120, (300, 200, 3), dtype=np.uint8)
        cv2.rectangle(img, (80, 50), (120, 250), (150, 150, 150), -1)  # Vertical person
        
    elif mode == 'fallen':
        # Horizontal person (wider than tall)
        img = np.random.randint(80, 120, (200, 400, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 80), (350, 120), (100, 100, 100), -1)  # Horizontal person
        
    elif mode == 'leaning':
        # Diagonal/leaning person
        img = np.random.randint(80, 120, (300, 250, 3), dtype=np.uint8)
        # Draw diagonal rectangle to simulate leaning
        points = np.array([[60, 50], [180, 80], [170, 250], [50, 220]], np.int32)
        cv2.fillPoly(img, [points], (130, 130, 130))
        
    return img

def demonstrate_model_features():
    """Demonstrate what the model learned"""
    print("\n🔍 Model Feature Analysis")
    print("=" * 30)
    
    model = load_model('fall_model.json')
    if model is None:
        print("❌ No model found")
        return
    
    print("📋 Learned Features:")
    for feature_name, params in model.items():
        direction = "Higher" if params['higher_indicates_fall'] else "Lower"
        print(f"   {feature_name}:")
        print(f"     Fall avg: {params['fall_mean']:.3f}")
        print(f"     Normal avg: {params['normal_mean']:.3f}")
        print(f"     Threshold: {params['threshold']:.3f}")
        print(f"     Rule: {direction} values indicate fall risk")
        print()

def main():
    """Run all tests"""
    print("🏥 Fall Detection Model Testing Suite")
    print("=" * 45)
    
    # Test the model
    success = test_trained_model()
    
    # Show model analysis
    demonstrate_model_features()
    
    if success:
        print("\n🎉 Model testing completed successfully!")
        print("\n💡 Integration ready:")
        print("   The model can now be integrated into fall_detection.py")
        print("   for real-time fall detection enhancement.")
    else:
        print("\n⚠️ Model needs improvement - consider retraining")

if __name__ == "__main__":
    main()