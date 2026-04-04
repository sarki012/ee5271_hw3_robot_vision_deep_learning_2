# EE 5271 Robot Vision Homework 3 Deep Learning
# Erik Sarkinen
# 3854563
# 4/1/26

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import scipy.io
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class SLPLinear(nn.Module):
    def __init__(self, input_size=196, num_classes=10):
        super(SLPLinear, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
    def forward(self, x):
        return self.fc(x)
    
class SLP(nn.Module):
    def __init__(self, input_size=196, num_classes=10):
        super(SLP, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
    def forward(self, x):
        return self.fc(x)

class MLP(nn.Module):
    def __init__(self, input_size=196, hidden_size=30,num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
        out_channels=3,
        kernel_size=3,
        stride=1,
        padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(3 * 7 * 7, num_classes)
    def forward(self, x):
        # Reshape to (batch, 1, 14, 14)
        if len(x.shape) == 2:
            x = x.view(-1, 1, 14, 14)
        x = self.conv1(x) #-> (batch, 3, 14, 14)
        x = self.relu(x)
        x = self.pool(x) #-> (batch, 3, 7, 7)
        x = x.view(x.size(0),-1) #-> (batch, 147)
        x = self.fc(x)
        #-> (batch, 10)
        return x
        
        
        