from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

train_data = np.loadtxt('./data/train_data.csv',delimiter=',',dtype="int")
train_label= np.loadtxt('./data/train_label.csv',delimiter=',',dtype="int")
test_data = np.loadtxt('./data/test_data.csv',delimiter=',',dtype="int")

class HandWritten(Dataset):
    def __init__(self, address, transform):
        self.transform = transform
        self.data = np.loadtxt(address,delimiter=',',dtype="int")
    
    def __len__():
        return data.shape[0]
    
    def __getitem(self, idx):
        if self.transform:
            self.data[idx] = self.transform(data[idx])

        return data[idx]




        self.transform = transform
        self.data = pd.read_csv(adata)
        self.data = self.data.values().tolist()
        self.data = np.array(self.data)
        self.data = self.data.reshape(self.data.shape[0], 1,64, 64)
        self.label = pd.read_cvs(alabel)