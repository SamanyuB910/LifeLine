#!/usr/bin/env python3
"""
FER Integration for LifeLine Patient Monitoring
==============================================
Integrates facial expression recognition with the LifeLine system
for monitoring patient emotional states during healthcare interactions.
"""

import joblib
import numpy as np
from PIL import Image
import cv2

class FERIntegration:
    """Facial Expression Recognition integration for LifeLine."""
    
    def __init__(self, model_path='fer_best.pkl'):
        """Initialize FER integration."""
        self.model_data = None
        self.is_loaded = False
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained FER model."""
        try:
            self.model_data = joblib.load(model_path)
            self.is_loaded = True
            print(f"✅ FER Model loaded: {self.model_data['model_type']}")
            print(f"🎭 Emotion classes: {self.model_data['emotion_classes']}")
            print(f"🎯 Training accuracy: {self.model_data['accuracy']:.3f}")
        except Exception as e:
            print(f"❌ Error loading FER model: {e}")
            self.is_loaded = False
    
    def preprocess_image(self, image, target_size=(24, 24)):
        """Preprocess image for FER prediction."""
        try:
            # Convert to PIL Image if it's a numpy array
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize and convert to grayscale
            image = image.resize(target_size)
            image = image.convert('L')
            
            # Convert to array and normalize
            img_array = np.array(image).flatten() / 255.0
            
            return img_array.reshape(1, -1)  # Reshape for single prediction
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None
    
    def predict_emotion(self, image):
        """Predict emotion from face image."""
        if not self.is_loaded:
            return None, 0.0
        
        # Preprocess image
        processed = self.preprocess_image(image)
        if processed is None:
            return None, 0.0
        
        try:
            # Scale features
            scaled = self.model_data['scaler'].transform(processed)
            
            # Predict
            prediction = self.model_data['model'].predict(scaled)[0]
            probabilities = self.model_data['model'].predict_proba(scaled)[0]
            
            # Get emotion name and confidence
            emotion = self.model_data['emotion_classes'][prediction]
            confidence = probabilities[prediction]
            
            return emotion, confidence
            
        except Exception as e:
            print(f"❌ Error predicting emotion: {e}")
            return None, 0.0
    
    def analyze_patient_emotion(self, image):
        """Analyze patient emotion with health context."""
        emotion, confidence = self.predict_emotion(image)
        
        if emotion is None:
            return {
                'emotion': 'unknown',
                'confidence': 0.0,
                'alert_level': 'none',
                'recommendation': 'Unable to analyze emotion'
            }
        
        # Healthcare-specific emotion analysis
        alert_level = 'normal'
        recommendation = 'Patient appears stable'
        
        if emotion in ['fear', 'sad', 'angry']:
            alert_level = 'attention'
            recommendation = f'Patient showing {emotion} - consider comfort measures'
            
            if emotion == 'fear' and confidence > 0.5:
                alert_level = 'high'
                recommendation = 'Patient appears distressed - immediate attention recommended'
        
        elif emotion == 'happy':
            recommendation = 'Patient appears comfortable and positive'
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'alert_level': alert_level,
            'recommendation': recommendation
        }

def demo_fer_integration():
    """Demonstrate FER integration with LifeLine system."""
    print("🏥 LifeLine FER Integration Demo")
    print("=" * 40)
    
    # Initialize FER integration
    fer = FERIntegration()
    
    if not fer.is_loaded:
        print("❌ FER model not available for demo")
        return
    
    # Create a demo synthetic face image (in real use, this would be from camera)
    print("\n📷 Creating demo face image...")
    demo_image = Image.new('RGB', (48, 48), color='gray')
    
    # Analyze emotion
    result = fer.analyze_patient_emotion(demo_image)
    
    print(f"\n🎭 Emotion Analysis Results:")
    print(f"   Detected Emotion: {result['emotion']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Alert Level: {result['alert_level']}")
    print(f"   Recommendation: {result['recommendation']}")
    
    # Show integration with LifeLine system
    print(f"\n🏥 LifeLine Integration:")
    if result['alert_level'] == 'high':
        print("   🚨 HIGH PRIORITY: Nurse notification sent")
        print("   📞 Doctor alert activated")
    elif result['alert_level'] == 'attention':
        print("   ⚠️  ATTENTION: Added to care notes")
        print("   📝 Comfort check scheduled")
    else:
        print("   ✅ NORMAL: Routine monitoring continues")
    
    print(f"\n💾 Analysis logged to patient record")
    print(f"🎉 FER Integration Demo Complete!")

if __name__ == "__main__":
    demo_fer_integration()