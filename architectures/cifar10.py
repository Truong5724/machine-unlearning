import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()

        # Load pretrained ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Remove original fully connected layer
        in_features = self.backbone.fc.in_features

        # Custom classification head
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)   