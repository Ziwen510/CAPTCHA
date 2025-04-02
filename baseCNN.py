import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseCNN(nn.Module):
    def __init__(self, num_classes=37, hidden_channels=64, input_channels=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.classifier_conv = nn.Conv2d(hidden_channels, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)  # (B, 32, H/2, W/2)
        x = self.conv2(x)  # (B, 64, H/4, W/4)
        x = self.conv3(x)  # (B, hidden_channels, H/8, W/8)
        x = self.conv4(x)  # (B, hidden_channels, H/8, W/8)
        x = self.classifier_conv(x)  # (B, num_classes, H/8, W/8)
        x = x.mean(dim=2)  # (B, num_classes, W/8)
        return x
