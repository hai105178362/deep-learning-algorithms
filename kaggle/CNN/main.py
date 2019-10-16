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
import cnnmodel as M
from cnn_params import num_feats, hidden_sizes, num_classes, learningRate, weightDecay, device, train_dataloader, dev_dataloader,lr_cent,feat_dim

if __name__ == "__main__":
    network = M.network
    network.apply(M.init_weights)
    network.train()
    network.to(device)
    M.train_closs(network, train_dataloader, dev_dataloader)
