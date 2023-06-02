import torch.nn as nn
import torch

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(38400, 128)  # Adjusted input size for 100x50 images
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.relu1(x1)
        x1 = self.maxpool1(x1)
        x1 = self.conv2(x1)
        x1 = self.relu2(x1)
        x1 = self.maxpool2(x1)
        x1 = x1.view(x1.size(0), -1)
        
        x2 = self.conv1(x2)
        x2 = self.relu1(x2)
        x2 = self.maxpool1(x2)
        x2 = self.conv2(x2)
        x2 = self.relu2(x2)
        x2 = self.maxpool2(x2)
        x2 = x2.view(x2.size(0), -1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
        