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
from data_utils import get_test_loader, get_train_loader

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            labels_onehot = F.one_hot(labels.long(), num_classes=10)
            loss = criterion(outputs, labels_onehot)
            
            loss.backward()
            optimizer.step()
        print(loss.item())
        scheduler.step() # Update learning rate
        test_loader = get_test_loader()
        # Evaluate on test set
        test_acc = evaluate(model, test_loader, device='cpu')
        print(f'Epoch {epoch+1}: Test Acc = {test_acc:.2f}%')

def evaluate(model, test_loader, device='cpu'):       #
    # 1. Set model to evaluation mode
    model.eval()
    correct = 0
    total = 0
    # 2. Disable gradient calculation to save memory and speed up
    with torch.no_grad():
        for data, labels_onehot in test_loader:
            data, labels_onehot = data.to(device), labels_onehot.to(device)
            # 3. Forward pass
            outputs = model(data)
            # 4. Get predictions (index of max log-probability)
            _, predicted = torch.max(outputs.data, 1)
            
            # FIX: Convert one-hot labels to class indices
            labels = torch.argmax(labels_onehot, dim=1) 
            # 5. Aggregate results
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 
             
    # 6. Calculate final accuracy
    test_acc = 100 * correct / total
    return test_acc

          #  labels_onehot = torch.zeros(labels.size(0), 10)  
         #   labels_onehot.scatter_(1, labels, 1)   
            # Use the outputs from the current batch, not 'model.outputs'
            # FIXED: Proper one-hot encoding using built-in function