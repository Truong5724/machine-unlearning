import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskModel(nn.Module):
    def __init__(self, input_shape=(3, 64, 64), dropout_rate=0.3):
        super().__init__()

        channels = input_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.flat_size = 128 * 8 * 8
        self.shared_fc = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.head_gender = nn.Linear(128, 2)
        self.head_age = nn.Linear(128, 5)
        self.head_race = nn.Linear(128, 5)

    def _encode(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.shared_fc(x)
        return x

    def forward(self, x):
        h = self._encode(x)
        return {
            "gender": self.head_gender(h),
            "age": self.head_age(h),
            "race": self.head_race(h),
        }

    def forward_task(self, x, task):
        h = self._encode(x)
        if task == "gender":
            return self.head_gender(h)
        if task == "age":
            return self.head_age(h)
        if task == "race":
            return self.head_race(h)
        raise ValueError(f"Unsupported task: {task}")
