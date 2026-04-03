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
        x = torch.flatten(x, 1)
        print(f"Input shape: {x.shape}") # Should be (batch_size, 196)
        return self.fc(x)   
    
'''
class SLPLinear(nn.Module):
    def __init__(self, input_size=196, num_classes=10):
        super(SLPLinear, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

'''
