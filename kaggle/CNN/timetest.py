import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import sys
import torch
import torchvision





if __name__ == "__main__":
    test_dataset = torchvision.datasets.ImageFolder(root='data.nosync/validation_classification/medium', transform=torchvision.transforms.ToTensor())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)
    label_ref = [(i,j) for i,j in enumerate(test_dataset.classes)]
    model.load_state_dict(torch.load('saved_models/cnn_epoch6.pt', map_location=device))
