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

    '''
    # 3. Convert numpy arrays to PyTorch tensors
    X_train = torch.tensor(images_train, dtype=torch.float32)
    X_train = X_train.t()
    y_train = torch.tensor(labels_train, dtype=torch.long) # Use torch.long for classification labels
    y_train = y_train.t()
    '''
    # Assuming images_train was 784x60000
    X_train = torch.tensor(labels_train, dtype=torch.float32).t() # Now 60000x784
    #y_train = torch.tensor(images_train, dtype=torch.long).reshape(-1) # Now 60000
    y_train = torch.tensor(images_train, dtype=torch.long).t()#.squeeze()

    print(X_train.size())
    print(y_train.size())

    train_dataset = TensorDataset(X_train, y_train)
    # You can also create a DataLoader to handle batching and shuffling
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    # 2. Extract the specific image (e.g., first column) and reshape
    # Check shape first: print(image_data.shape)
 #   plt.imshow(images_train[:,130].reshape((14, 14), order='F'), cmap='gray')
  #  plt.show()
    return train_loader

def get_test_loader():
    mat_data_test = scipy.io.loadmat('../../resource/hw3/data/mnist_test.mat')
    images_test = mat_data_test['im_test']
    labels_test = mat_data_test['label_test']

    '''
    X_test = torch.tensor(images_test, dtype=torch.float32)
    X_test = X_test.t()
    y_test = torch.tensor(labels_test, dtype=torch.long)
    y_test = y_test.t()
    '''
   # y_test_normalized = F.normalize(y_test, p=2, dim=1) # L2 norm
      # Assuming images_train was 784x60000
    X_test = torch.tensor(labels_test, dtype=torch.float32).t() # Now 60000x784
    #y_train = torch.tensor(images_train, dtype=torch.long).reshape(-1) # Now 60000
    y_test = torch.tensor(images_test, dtype=torch.long).squeeze()
 #   X_test = torch.tensor(labels_test, dtype=torch.float32).t() # Now 60000x784
  #  y_test = torch.tensor(images_test, dtype=torch.long).view(-1) # Now 60000
 #   print(X_test.size())
 #   print(y_test.size())
   # test_dataset = TensorDataset(X_test, y_test_normalized)
    test_dataset = TensorDataset(X_test, y_test)
    # You can also create a DataLoader to handle batching and shuffling
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)
   # plt.imshow(images_test[:,123].reshape((14, 14), order='F'), cmap='gray')
  #  plt.show()
    return test_loader

    #y_train = torch.tensor(images_train, dtype=torch.long).view(-1) # Now 60000
    # 3. Alternative: Norm along a specific dimension (e.g., norm of each image)
    # This calculates the norm of each of the 60,000 images (dim 0 or 1 depending on t())
  #  image_norms = torch.norm(X_train, dim=1) 
 #   print(f"Shape of image norms: {image_norms.shape}")
 #   result = torch.norm(my_tensor.float())
    #y_train_normalized = F.normalize(y_train, p=2, dim=1) # L2 norm
#    print(X_train.size())
 #   print(y_train.size())
    # Now X_train and y_train are defined and can be used in TensorDataset
    #train_dataset = TensorDataset(X_train, y_train_normalized)