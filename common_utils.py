### THIS FILE CONTAINS COMMON FUNCTIONS, CLASSSES

import tqdm
import time
import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from scipy.io import wavfile as wav

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix



def split_dataset(df, columns_to_drop, test_size, random_state):
    label_encoder = preprocessing.LabelEncoder()

    df['label'] = label_encoder.fit_transform(df['label'])

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    df_train2 = df_train.drop(columns_to_drop,axis=1)
    y_train2 = df_train['label'].to_numpy()

    df_test2 = df_test.drop(columns_to_drop,axis=1)
    y_test2 = df_test['label'].to_numpy() 

    return df_train2, y_train2, df_test2, y_test2

def preprocess_dataset(df_train, df_test):

    standard_scaler = preprocessing.StandardScaler()
    df_train_scaled = standard_scaler.fit_transform(df_train)

    df_test_scaled = standard_scaler.transform(df_test)

    return df_train_scaled, df_test_scaled

def set_seed(seed = 0):
    '''
    set random seed
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Define the preprocessing function
def preprocess(df, col_drop):
    # Columns to drop
    columns_to_drop = col_drop
    
    # Split the dataset (80% train, 20% test)
    X_train, y_train, X_test, y_test = split_dataset(df, columns_to_drop, test_size=0.2, random_state=1)

    # Scale the features using StandardScaler
    X_train_scaled, X_test_scaled = preprocess_dataset(X_train, X_test)
    
    return X_train_scaled, y_train, X_test_scaled, y_test

    
# early stopping obtained from tutorial
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



class MLP(nn.Module):
    
    def __init__(self, no_features, no_hidden=128, no_labels=1, depth = 2):
        super(MLP, self).__init__()
        
        # Define the MLP stack using nn.Sequential
        self.mlp_stack = nn.Sequential(
            # First hidden layer
            nn.Linear(no_features, no_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Apply dropout to the first hidden layer

            # Second hidden layer
            nn.Linear(no_hidden, no_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Apply dropout to the second hidden layer

            # Third hidden layer
            nn.Linear(no_hidden, no_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Apply dropout to the third hidden layer

            # Output layer
            nn.Linear(no_hidden, no_labels),
            nn.Sigmoid()  # Sigmoid activation for the output layer
        )

    def forward(self, x):
        # Pass the input through the network stack
        return self.mlp_stack(x)


# Define the Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X (ndarray): The input features (e.g., X_train_scaled or X_test_scaled).
            y (ndarray): The labels (e.g., y_train or y_test).
        """
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert input features to PyTorch tensors
        self.y = torch.tensor(y, dtype=torch.long)     # Convert labels to PyTorch tensors (long for classification)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to fetch.
        Returns:
            Tuple of (input features, label) for the given index.
        """
        return self.X[idx], self.y[idx]

