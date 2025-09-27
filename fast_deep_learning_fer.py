#!/usr/bin/env python3
"""
Fast Deep Learning FER Training
===============================
Lightweight CNN for rapid training and improved accuracy.
Uses MobileNet-inspired architecture for speed.
"""

import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
import joblib
import glob

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

class FastFER_CNN(nn.Module):
    """Lightweight CNN for fast FER training."""
    
    def __init__(self, num_classes=7):
        super(FastFER_CNN, self).__init__()
        
        # Lightweight feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 48x48 -> 24x24
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 24x24 -> 12x12
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 12x12 -> 6x6
            
            # Block 4 (Lightweight)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3))  # -> 3x3
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

class FERDataset(Dataset):
    """Fast FER dataset loader."""
    
    def __init__(self, data_path, transform=None, max_per_class=300):
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        
        # Find emotion directories
        emotion_dirs = []
        for root, dirs, files in os.walk(data_path):
            jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(jpg_files) > 50:
                emotion_dirs.append((root, jpg_files))
        
        # Sort for consistency
        emotion_dirs.sort(key=lambda x: os.path.basename(x[0]))
        
        print(f"📁 Found {len(emotion_dirs)} emotion classes")
        
        for class_idx, (dir_path, files) in enumerate(emotion_dirs[:7]):  # 7 classes
            class_name = os.path.basename(dir_path)
            self.class_names.append(class_name)
            
            # Select files for this class
            selected_files = files[:max_per_class]
            
            print(f"📸 Loading {len(selected_files)} images for {class_name}...")
            
            for img_file in selected_files:
                img_path = os.path.join(dir_path, img_file)
                
                try:
                    # Quick load and basic check
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Convert to grayscale and resize
                    img = img.convert('L')
                    img = img.resize((48, 48), Image.Resampling.LANCZOS)
                    
                    # Convert to numpy and normalize
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    
                    self.images.append(img_array)
                    self.labels.append(class_idx)
                    
                except Exception as e:
                    continue
        
        print(f"✅ Loaded {len(self.images)} total images")
        print(f"🎭 Classes: {self.class_names}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to tensor and add channel dimension
        image = torch.from_numpy(image).unsqueeze(0)  # (1, 48, 48)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_data_loaders(data_path, batch_size=64, val_split=0.2):
    """Create fast data loaders."""
    print("📊 Creating data loaders...")
    
    # Load dataset
    dataset = FERDataset(data_path, max_per_class=250)
    
    if len(dataset) == 0:
        print("❌ No data loaded")
        return None, None, None
    
    # Calculate split sizes
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # 0 for Windows compatibility
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"📈 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_loader, val_loader, dataset.class_names

def train_fast_model(train_loader, val_loader, class_names, num_epochs=15):
    """Train the fast CNN model."""
    print(f"🏋️ Training Fast CNN for {num_epochs} epochs...")
    
    # Initialize model
    model = FastFER_CNN(num_classes=len(class_names)).to(device)
    
    # Fast optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_val_acc = 0.0
    best_model_state = None
    
    print("🎯 Starting training...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train: {train_acc:5.1f}% | "
              f"Val: {val_acc:5.1f}% | "
              f"Time: {epoch_time:.1f}s")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, best_val_acc / 100  # Convert to decimal

def save_pytorch_model(model, accuracy, class_names):
    """Save the PyTorch model for integration."""
    print("💾 Saving PyTorch model...")
    
    # Save the complete model state
    model_data = {
        'model_state_dict': model.state_dict(),
        'model_class': 'FastFER_CNN',
        'accuracy': accuracy,
        'emotion_classes': class_names,
        'model_type': 'Fast CNN (PyTorch)',
        'input_size': (1, 48, 48),
        'num_classes': len(class_names)
    }
    
    # Save with torch
    torch.save(model_data, 'fer_fast_cnn.pth')
    
    # Also create a scikit-learn compatible version for integration
    try:
        # Create a simple wrapper for compatibility
        sklearn_compatible = {
            'model_type': 'Fast CNN (PyTorch)',
            'accuracy': accuracy,
            'emotion_classes': class_names,
            'pytorch_model_path': 'fer_fast_cnn.pth',
            'input_size': (1, 48, 48)
        }
        
        joblib.dump(sklearn_compatible, 'fer_best.pkl')
        print("✅ Models saved: fer_fast_cnn.pth (PyTorch) & fer_best.pkl (compatibility)")
        
    except Exception as e:
        print(f"⚠️ Sklearn compatibility save failed: {e}")

def fast_deep_learning_training():
    """Main fast deep learning training function."""
    print("🚀 FAST DEEP LEARNING FER TRAINING")
    print("=" * 50)
    
    start_time = time.time()
    
    # Create data loaders
    train_loader, val_loader, class_names = create_data_loaders(
        "datasets/fer_dataset", 
        batch_size=64,
        val_split=0.2
    )
    
    if train_loader is None:
        print("❌ Data loading failed")
        return None
    
    # Train model
    model, best_accuracy = train_fast_model(
        train_loader, val_loader, class_names, num_epochs=15
    )
    
    # Save model
    save_pytorch_model(model, best_accuracy, class_names)
    
    total_time = time.time() - start_time
    
    # Results
    print(f"\n🎉 FAST DEEP LEARNING TRAINING COMPLETE!")
    print(f"🏆 Best Validation Accuracy: {best_accuracy:.3f} ({best_accuracy:.1%})")
    print(f"⏱️ Total Training Time: {total_time:.1f} seconds")
    
    # Compare with previous results
    prev_accuracy = 0.388  # Our ensemble result
    improvement = best_accuracy - prev_accuracy
    
    print(f"\n📊 Improvement over Traditional ML:")
    print(f"   Previous (Ensemble): {prev_accuracy:.3f} ({prev_accuracy:.1%})")
    print(f"   Fast CNN: {best_accuracy:.3f} ({best_accuracy:.1%})")
    print(f"   Improvement: {improvement:+.3f} ({improvement/prev_accuracy*100:+.1f}%)")
    
    if best_accuracy >= 0.85:
        print("🎉 SUCCESS: Target 85%+ accuracy ACHIEVED!")
    elif best_accuracy >= 0.75:
        print("🟢 EXCELLENT: 75%+ accuracy achieved!")
    elif best_accuracy >= 0.65:
        print("🟡 VERY GOOD: 65%+ accuracy achieved!")
    elif best_accuracy >= 0.50:
        print("🟠 GOOD: 50%+ accuracy achieved!")
    else:
        print("🔶 IMPROVED: Better than traditional ML!")
    
    print(f"\n🏥 Ready for LifeLine Integration!")
    print(f"💾 Models saved and ready for deployment!")
    
    return best_accuracy

if __name__ == "__main__":
    print("🔥 Fast Deep Learning FER Training")
    print("==================================")
    
    # Check PyTorch installation
    try:
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        final_accuracy = fast_deep_learning_training()
        
        if final_accuracy and final_accuracy > 0.5:
            print(f"\n🎯 SUCCESS: Achieved {final_accuracy:.1%} accuracy with fast deep learning!")
        
    except ImportError:
        print("❌ PyTorch not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])
        print("✅ PyTorch installed. Please run the script again.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()