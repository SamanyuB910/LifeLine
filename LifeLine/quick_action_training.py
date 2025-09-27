# quick_action_training.py - Quick action recognition training
import sys
sys.path.append('.')
from lightweight_action_recognition import SimplePoseDataset, ActionLSTM, train_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import json

def quick_train():
    """Quick training with reduced epochs"""
    print("🏋️ Quick Action Recognition Training")
    print("=" * 40)
    
    # Configuration
    SEQUENCE_LENGTH = 8  # Reduced for speed
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    EPOCHS = 5  # Reduced for quick training
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🖥️ Using device: {DEVICE}")
    
    # Create dataset
    dataset = SimplePoseDataset(sequence_length=SEQUENCE_LENGTH)
    
    # Split dataset
    train_indices, val_indices = train_test_split(
        range(len(dataset)), 
        test_size=0.2, 
        random_state=42,
        stratify=dataset.encoded_labels
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)} samples")
    
    # Model
    model = ActionLSTM(
        input_size=20,
        hidden_size=64,
        num_layers=2,
        num_classes=dataset.num_classes,
        dropout=0.2
    ).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_acc = 0.0
    
    print(f"\n🚀 Training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train: {avg_train_loss:.4f}/{train_acc:5.1f}% | "
              f"Val: {avg_val_loss:.4f}/{val_acc:5.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'num_classes': dataset.num_classes,
                'action_classes': dataset.label_encoder.classes_.tolist(),
                'model_config': {
                    'input_size': 20,
                    'hidden_size': 64,
                    'num_layers': 2,
                    'sequence_length': SEQUENCE_LENGTH
                }
            }, 'weizmann_pose_model.pth')
    
    print(f"\n🎉 Quick training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.1f}%")
    print(f"💾 Model saved: weizmann_pose_model.pth")
    
    print(f"\n🎯 Enhanced Fall Detection Ready:")
    print(f"   ✅ Temporal sequence analysis: TRAINED")
    print(f"   ✅ Multi-keypoint tracking: TRAINED")
    print(f"   ✅ Movement patterns: TRAINED")
    print(f"   ✅ Action classification: TRAINED")
    
    return model, dataset, best_val_acc

if __name__ == "__main__":
    try:
        quick_train()
        print("\n🚀 Action recognition training complete!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()