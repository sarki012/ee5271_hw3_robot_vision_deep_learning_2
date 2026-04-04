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
from models import SLPLinear, SLP, MLP, CNN
from torch.optim.lr_scheduler import StepLR
from data_utils import get_train_loader
import train

def main():
    
    # Instantiate the class
    model = SLPLinear(input_size=196, num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    # Define the StepLR scheduler
    # step_size: how many epochs to wait before reducing the LR (10)
    # gamma: the multiplier for the LR (0.9)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    num_epochs = 30
    criterion = nn.MSELoss()
    train_loader = get_train_loader()
    train.train_model_SLPLinear(model, train_loader, criterion, optimizer, scheduler, num_epochs)

    # Instantiate the class
    model = SLP(input_size=196, num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # Define the StepLR scheduler
    # step_size: how many epochs to wait before reducing the LR (10)
    # gamma: the multiplier for the LR (0.95)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.95)
    num_epochs = 30
    criterion = nn.CrossEntropyLoss()
    train_loader = get_train_loader()
    train.train_modelSLP(model, train_loader, criterion, optimizer, scheduler, num_epochs)

    # Instantiate the class
    model = MLP(input_size=196, num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # Define the StepLR scheduler
    # step_size: how many epochs to wait before reducing the LR (10)
    # gamma: the multiplier for the LR (0.95)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.95)
    num_epochs = 30
    criterion = nn.CrossEntropyLoss()
    train_loader = get_train_loader()
    train.train_modelMLP(model, train_loader, criterion, optimizer, scheduler, num_epochs)

    # Instantiate the class
    model = CNN(num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    # Define the StepLR scheduler
    # step_size: how many epochs to wait before reducing the LR (10)
    # gamma: the multiplier for the LR (0.95)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.95)
    num_epochs = 30
    criterion = nn.CrossEntropyLoss()
    train_loader = get_train_loader()
    train.train_modelCNN(model, train_loader, criterion, optimizer, scheduler, num_epochs)

if __name__ == "__main__":
    main()




