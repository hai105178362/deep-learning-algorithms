import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import decode
import helper.phoneme_list as PL

valypath = "../dataset.nosync/HW3P2_Data/wsj0_dev_merged_labels.npy"
y = np.load(valypath, allow_pickle=True, encoding="bytes")
print(y[0])
result = ""
for i in y[0]:
    result += PL.PHONEME_MAP[i]
print(result)