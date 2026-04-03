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
    return train_loader

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
    return test_loader