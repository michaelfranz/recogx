# Here we define a CNN model with three convolutional layers, two fully connected layers, and ReLU activation
# functions. The model takes in Mel-frequency cepstrum (MFC) images of size 98x12 and outputs a binary classification
# (male or female).

import torch
import torch.nn as nn


class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1472, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 1472)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
