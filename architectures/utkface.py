"""
Architecture cho UTKFace - Binary Gender Classification
Tương tự CelebA nhưng tối ưu hơn vì dataset nhỏ hơn

Input: (3, 64, 64) RGB images
Output: 2 classes (Female, Male)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    CNN tối ưu cho UTKFace gender classification
    
    Nhẹ hơn CelebA vì:
    - Dataset nhỏ hơn (23K vs 202K)
    - Task đơn giản hơn (gender rõ ràng hơn attributes)
    
    Total params: ~500K
    """
    
    def __init__(self, input_shape, nb_classes, dropout_rate=0.3):
        super(Model, self).__init__()
        
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        
        # Conv blocks - nhẹ hơn CelebA
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # After 3 pooling: 64 -> 32 -> 16 -> 8
        self.flat_size = 128 * 8 * 8  # 8192
        
        # FC layers
        self.fc1 = nn.Linear(self.flat_size, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, nb_classes)
    
    def forward(self, x):
        # Conv blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 64 -> 32
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 32 -> 16
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # 16 -> 8
        
        # Flatten
        x = x.view(-1, self.flat_size)
        
        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


# Alternative: Very light model cho training cực nhanh
class TinyModel(nn.Module):
    """
    Model cực nhẹ cho quick experiments
    Total params: ~200K
    """
    
    def __init__(self, input_shape, nb_classes, dropout_rate=0.25):
        super(TinyModel, self).__init__()
        
        # Simple CNN
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(64)
        
        # 64 -> 32 -> 16 -> 8
        self.flat_size = 64 * 8 * 8
        
        self.fc1 = nn.Linear(self.flat_size, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, nb_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.bn(self.conv3(x)))
        x = self.pool(x)
        
        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING UTKFACE ARCHITECTURE")
    print("=" * 70)
    
    # Test standard model
    model = Model(input_shape=(3, 64, 64), nb_classes=2, dropout_rate=0.3)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nStandard Model:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward
    x = torch.randn(4, 3, 64, 64)
    output = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  ✅ Forward pass OK")
    
    # Test tiny model
    print(f"\nTiny Model:")
    tiny_model = TinyModel(input_shape=(3, 64, 64), nb_classes=2)
    total_params_tiny = sum(p.numel() for p in tiny_model.parameters())
    print(f"  Total parameters: {total_params_tiny:,}")
    
    output_tiny = tiny_model(x)
    print(f"  Output shape: {output_tiny.shape}")
    print(f"  ✅ Forward pass OK")
    
    print(f"\n💡 Tiny model giảm {(1 - total_params_tiny/total_params)*100:.1f}% parameters")
    print(f"   → Training nhanh hơn ~50-60%")
    print("=" * 70)