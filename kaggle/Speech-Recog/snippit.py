import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import time
import pandas as pd
import csv

# source = torch.rand((5,10))
# # now we expand to size (7, 11) by appending a row of 0s at pos 0 and pos 6,
# # and a column of 0s at pos 10
# result = F.pad(input=source, pad=(0, 0, 2, 2), mode='constant', value=0)
# print(result)

# source = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
# # result = np.pad(source, ((1, 1), (0, 0)), 'constant', constant_values=0)
#
# result = np.pad(source, ((1, 1), (0, 0)),'reflect', reflect_type='odd')
# print(result)
#################################
a = pd.read_csv('tmpresult_2.csv')
b = pd.read_csv('devref.csv')
correct = 0
arr1, arr2 = a['label'], b['label']
for i, j in zip(arr1, arr2):
    assert len(arr1) == len((arr2)), 'Output has diff lenght'
    if i == j:
        correct += 1
print(correct / len(arr1))

##################
# trainy = np.load("source_data.nosync/dev_labels.npy", allow_pickle=True)
# dev_label = []
# for i in trainy:
#     dev_label.extend(i)
#
# with open('devref.csv', mode='w') as csv_file:
#     fieldnames = ['id', 'label']
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#     writer.writeheader()
#     for i in range(len(dev_label)):
#         writer.writerow({'id': i, 'label': int(dev_label[i])})
# print(len(dev_label))