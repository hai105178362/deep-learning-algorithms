import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(xpath, ypath=None):
    x = np.load(xpath, allow_pickle=True, encoding="bytes")
    if ypath != None:
        y = np.load(ypath, allow_pickle=True, encoding="bytes")
        return x, y
    return x


class LinesDataset(Dataset):
    def __init__(self, lines):
        self.lines = [torch.tensor(l) for l in lines]

    def __getitem__(self, i):
        line = self.lines[i]
        return line[:-1].to(DEVICE), line[1:].to(DEVICE)

    def __len__(self):
        return len(self.lines)


# collate fn lets you control the return value of each batch
# for packed_seqs, you want to return your data sorted by length
def collate_lines(seq_list):
    inputs, targets = zip(*seq_list)
    lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i] for i in seq_order]
    targets = [targets[i] for i in seq_order]
    return inputs, targets


# l = DataLoader()


devxpath = "dataset.nosync/HW3P2_Data/wsj0_dev.npy"
devypath = "dataset.nosync/HW3P2_Data/wsj0_dev_merged_labels.npy"
trainxpath = "dataset.nosync/HW3P2_Data/wsj0_train.npy"
trainypath = "dataset.nosync/HW3P2_Data/wsj0_train_merged_labels.npy"
task = "dev"
if task == "train":
    xpath = trainxpath
    ypath = trainypath
else:
    xpath = devxpath
    ypath = devypath
X, Y = load_data(xpath, ypath)
X = LinesDataset(X)
