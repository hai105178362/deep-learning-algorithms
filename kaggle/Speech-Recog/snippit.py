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

# source = torch.rand((5,10))
# # now we expand to size (7, 11) by appending a row of 0s at pos 0 and pos 6,
# # and a column of 0s at pos 10
# result = F.pad(input=source, pad=(0, 0, 2, 2), mode='constant', value=0)
# print(result)

source = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
result = np.pad(source, ((1, 1), (0, 0)), 'constant', constant_values=0)
print(result)
