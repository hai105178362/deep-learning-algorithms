import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import time
import net as net
import helper.phoneme_list as PL
import helper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


