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
import torch.optim as optim
from models import SLPLinear
from torch.optim.lr_scheduler import StepLR
import train
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, TensorDataset

# Load the .mat file
# 'file_path.mat' should be the path to your MATLAB file
mat_data_train = scipy.io.loadmat('../../resource/hw3/data/mnist_train.mat')
mat_data_test = scipy.io.loadmat('../../resource/hw3/data/mnist_test.mat')

# Print the keys to see the variable names inside the .mat file
print(mat_data_train.keys())


# Access specific variables (e.g., 'X' for images, 'y' for labels)
# Note: Variable names depend on how the .mat file was saved
images_train = mat_data_train['im_train']
labels_train = mat_data_train['label_train']

images_test = mat_data_test['im_test']
labels_test = mat_data_test['label_test']
#####################################

# 3. Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(images_train, dtype=torch.float32)
X_train = X_train.t()
y_train = torch.tensor(labels_train, dtype=torch.long) # Use torch.long for classification labels
y_train = y_train.t()
X_test = torch.tensor(images_test, dtype=torch.float32)
X_test = X_test.t()
y_test = torch.tensor(labels_test, dtype=torch.long)
y_test = y_test.t()

print(X_train.size())
print(y_train.size())

print(X_test.size())
print(y_test.size())

# Now X_train and y_train are defined and can be used in TensorDataset
train_dataset = TensorDataset(X_train, y_train)
# You can also create a DataLoader to handle batching and shuffling
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
# You can also create a DataLoader to handle batching and shuffling
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

# 2. Extract the specific image (e.g., first column) and reshape
# Check shape first: print(image_data.shape)
plt.imshow(images_train[:,130].reshape((14, 14), order='F'), cmap='gray')
plt.show()

plt.imshow(images_test[:,120].reshape((14, 14), order='F'), cmap='gray')
plt.show()

# Instantiate the class
model = SLPLinear(input_size=196, num_classes=10)
optimizer = optim.SGD(model.parameters(), lr=0.9)
# 2. Define the StepLR scheduler
# step_size: how many epochs to wait before reducing the LR (10)
# gamma: the multiplier for the LR (0.9)
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
num_epochs = 30
criterion = nn.MSELoss()

train.train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs)


