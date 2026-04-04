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
from data_utils import get_test_loader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    SLPLinear_path = "../../results/checkpoints/slp_linear.pth"
    test_acc = np.zeros((num_epochs), dtype=float)
    # 1. Initialize an empty list to store accuracies
    test_acc_history = []
    loss_history = [] 
    global count
    count = 0
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            labels_onehot = torch.zeros(labels.size(0), 10)  
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = labels_onehot.float()      
            # 3. Calculate loss
            # Use the outputs from the current batch, not 'model.outputs'
            loss = criterion(outputs, labels_onehot)
            loss.backward()
            optimizer.step()
        scheduler.step() # Update learning rate
        test_loader = get_test_loader()
        # Evaluate on test set
        test_acc = evaluate(model, test_loader, labels_onehot, device='cpu')
        test_acc_history.append(test_acc) # Store accuracy
        loss_history.append(loss.item())
        print(f'Epoch {epoch+1}: Test Acc = {test_acc:.2f}%')

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, SLPLinear_path)

    # 2. Plot after the loop finishes
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_history, label='Training Loss vs. Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epochs')
    plt.legend()
    plt.show() # Renders the plot

    plt.plot(range(1, num_epochs + 1), test_acc_history, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy vs. Epochs')
    plt.legend()
    plt.show() # Renders the plot

'''

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    MLP_path = "../../results/checkpoints/mlp.pth"
    test_acc = np.zeros((num_epochs), dtype=float)
    # 1. Initialize an empty list to store accuracies
    test_acc_history = [] 
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())     
            # 3. Calculate loss
            # Use the outputs from the current batch, not 'model.outputs'
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step() # Update learning rate
        test_loader = get_test_loader()
        # Evaluate on test set
        test_acc = evaluate(model, test_loader, labels, device='cpu')
        test_acc_history.append(test_acc) # Store accuracy
        loss_history.append(loss.item())
        print(f'Epoch {epoch+1}: Test Acc = {test_acc:.2f}%')

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, MLP_path)
'''


'''
    for i in range(num_epochs):
        # 2. Create the plot
        plt.plot(i, test_acc[i])
    # 3. Add labels and title (optional but recommended)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Loss Curve')
    # 4. Display the plot
    plt.show()
'''
def evaluate(model, test_loader, labels_onehot, device='cpu'):       #
    # 1. Set model to evaluation mode
    all_preds = []
    all_labels = []

    model.eval()
    correct = 0
    total = 0
    
    # 2. Disable gradient calculation to save memory and speed up
    with torch.no_grad():
        for data, labels_onehot in test_loader:
            data, labels_onehot = data.to(device), labels_onehot.to(device) 
            # 3. Forward pass
            outputs = model(data.float())
            # 4. Get predictions (index of max log-probability)
            _, predicted = torch.max(outputs.data, 1) 
            # 5. Aggregate results
            total += labels_onehot.size(0)
            correct += (predicted == labels_onehot).sum().item()  

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_onehot.cpu().numpy())
    # 6. Calculate final accuracy
    test_acc = 100 * correct / total
    # This will result in a 10x10 numpy array
    global count
    count += 1
    if count == 30:
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
        disp.plot(cmap=plt.cm.Blues)
        plt.title("MNIST Confusion Matrix")
        plt.show()
        count = 0
    return test_acc