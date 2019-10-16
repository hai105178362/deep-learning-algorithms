import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import time
import datetime
import cnnmodel

closs_weight = 1
lr_cent = 0.5
feat_dim = 10
numEpochs = 30
num_feats = 3
learningRate = 1e-2
weightDecay = 5e-5
hidden_sizes = [16, 32, 64, 144]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = torchvision.datasets.ImageFolder(root='data.nosync/validation_classification/medium', transform=torchvision.transforms.ToTensor())
# train_dataset = torchvision.datasets.ImageFolder(root='devset/medium', transform=torchvision.transforms.ToTensor())
dev_dataset = torchvision.datasets.ImageFolder(root='data.nosync/validation_classification/medium', transform=torchvision.transforms.ToTensor())
# dev_dataset = torchvision.datasets.ImageFolder(root='devset/medium_dev', transform=torchvision.transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=128, shuffle=True, num_workers=8)
num_classes = len(train_dataset.classes)
