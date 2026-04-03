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

def get_train_loader():
    mat_data_train = scipy.io.loadmat('../../resource/hw3/data/mnist_train.mat')
    # Assuming 'im_test' is (196, N) or (14, 14, N)
    images_train = mat_data_train['im_train']
    labels_train = mat_data_train['label_train']

    # 1. Flatten images to (N, 196) and transpose to (196, N) if needed
    # If images_test is (14, 14, N), reshape to (196, N)
    if len(images_train.shape) > 2:
        N = images_train.shape[-1]
        images_train = images_train.reshape((196, N), order='F') # Use 'F' for MATLAB column-major
    
    # 2. Correct transposition to ensure (196, N) or (N, 196)
    # Based on your code, you likely need (N, 196) for linear layers
    X_train = torch.tensor(images_train, dtype=torch.float32).t()
    y_train = torch.tensor(labels_train, dtype=torch.long).t()

    # If y_test is (N, 1) or (1, N), reshape it to a flat vector (N,)
    y_train = y_train.squeeze()                

    train_dataset = TensorDataset(X_train, y_train)                
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    
    # Verify shape
    # for images, labels in test_loader:
    #     print(images.shape) # Should be [32, 196]
    #     break                

    return train_loader

'''
# Load the .mat file
def get_train_loader():
    # 'file_path.mat' should be the path to your MATLAB file
    mat_data_train = scipy.io.loadmat('../../resource/hw3/data/mnist_train.mat')
    # Print the keys to see the variable names inside the .mat file
   # print(mat_data_train.keys())
    # Access specific variables (e.g., 'X' for images, 'y' for labels)
    # Note: Variable names depend on how the .mat file was saved
    images_train = mat_data_train['im_train']
    labels_train = mat_data_train['label_train']
    # 3. Convert numpy arrays to PyTorch tensors
    X_train = torch.tensor(labels_train, dtype=torch.float32)
    X_train = X_train.t()
    y_train = torch.tensor(images_train, dtype=torch.long) # Use torch.long for classification labels
    y_train = y_train.t()
  #  print(X_train.size())
   # print(y_train.size())
    # Now X_train and y_train are defined and can be used in TensorDataset
    train_dataset = TensorDataset(X_train, y_train)
    # You can also create a DataLoader to handle batching and shuffling
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    # 2. Extract the specific image (e.g., first column) and reshape
    # Check shape first: print(image_data.shape)
 #   plt.imshow(images_train[:,130].reshape((14, 14), order='F'), cmap='gray')
  #  plt.show()
    return train_loader
'''
def get_test_loader():
    mat_data_test = scipy.io.loadmat('../../resource/hw3/data/mnist_test.mat')
    # Assuming 'im_test' is (196, N) or (14, 14, N)
    images_test = mat_data_test['im_test']
    labels_test = mat_data_test['label_test']

    # 1. Flatten images to (N, 196) and transpose to (196, N) if needed
    # If images_test is (14, 14, N), reshape to (196, N)
    if len(images_test.shape) > 2:
        N = images_test.shape[-1]
        images_test = images_test.reshape((196, N), order='F') # Use 'F' for MATLAB column-major
    
    # 2. Correct transposition to ensure (196, N) or (N, 196)
    # Based on your code, you likely need (N, 196) for linear layers
    X_test = torch.tensor(images_test, dtype=torch.float32).t()
    y_test = torch.tensor(labels_test, dtype=torch.long).t()

    # If y_test is (N, 1) or (1, N), reshape it to a flat vector (N,)
    y_test = y_test.squeeze()                

    test_dataset = TensorDataset(X_test, y_test)                
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)
    
    # Verify shape
    # for images, labels in test_loader:
    #     print(images.shape) # Should be [32, 196]
    #     break                

    return test_loader

'''

def get_test_loader():
    mat_data_test = scipy.io.loadmat('../../resource/hw3/data/mnist_test.mat')
    images_test = mat_data_test['im_test']
    labels_test = mat_data_test['label_test']
    X_test = torch.tensor(labels_test, dtype=torch.float32)
    X_test = X_test.t()
    y_test = torch.tensor(images_test, dtype=torch.long)
    y_test = y_test.t()
  #  print(X_test.size())
  #  print(y_test.size())
    test_dataset = TensorDataset(X_test, y_test)
    # You can also create a DataLoader to handle batching and shuffling
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)
 #   plt.imshow(images_test[:,120].reshape((14, 14), order='F'), cmap='gray')
 #   plt.show()
    return test_loader
    '''