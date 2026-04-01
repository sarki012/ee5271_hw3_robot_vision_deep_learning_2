# EE 5271 Robot Vision Homework 3 Deep Learning
# Erik Sarkinen
# 3854563
# 4/1/26

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

# Create datasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
# Create data loaders
train_loader = DataLoader(train_dataset,
batch_size=32,
shuffle=True)
test_loader = DataLoader(test_dataset,
batch_size=32,
shuffle=False)