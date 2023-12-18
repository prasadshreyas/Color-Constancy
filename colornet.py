# Shreyas Prasad
# 10/18/23
# CS 7180: Advanced Perception

import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorNet(nn.Module):
    """
    ColorNet is a convolutional neural network model for color constancy.
    
    Args:
        num_channels (int): Number of input channels (default: 3)
    """
    def __init__(self, num_channels=3):
        super(ColorNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 240, kernel_size=(1, 1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(240)
        self.pool = nn.MaxPool2d(kernel_size=(8, 8), stride=(8, 8))
        self.fc1 = nn.Linear(4*4*240, 40)  # Adjusted to match the flattened size
        self.fc2 = nn.Linear(40, num_channels)

    def forward(self, x):
        """
        Forward pass of the ColorNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flattening to a 3840-element vector
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
