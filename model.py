# Here we define a CNN model with three convolutional layers, two fully connected layers, and ReLU activation
# functions. The model takes in Mel-frequency cepstrum (MFC) images of size 98x12 and outputs a binary classification
# (male or female).

import torch.nn as nn
import torch.nn.functional as F

class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()                   
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 8 * 1, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(nn.functional.relu(self.conv2(x)), 2)
        x = F.max_pool2d(nn.functional.relu(self.conv3(x)), 2)
        x = x.view(-1, 128 * 8 * 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
