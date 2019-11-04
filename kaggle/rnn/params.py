from torch.nn import CTCLoss
import torch
import torch.nn as nn
from torch.nn.utils.rnn import *
import numpy as np
import sys
import helper.phoneme_list as PL
import helper
from torch.utils import data
from torch.autograd import Variable

cuda = torch.cuda.is_available()
print("Cuda:{}".format(cuda))
device = torch.device("cuda" if cuda else "cpu")


def load_data(xpath, ypath=None):
    x = np.load(xpath, allow_pickle=True, encoding="bytes")
    if ypath != None:
        y = np.load(ypath, allow_pickle=True, encoding="bytes")
        return x, y
    return x





if __name__ == "__main__":
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
    print(len(X))
    # X_dataset = hel
    # word = ""
    # print(len(Y[1]))
    # for i in Y[0]:
    #     word += PL.PHONEME_MAP[i]
    # print(word)
    X_lens = torch.LongTensor([len(seq) for seq in X])
    Y_lens = torch.LongTensor([len(seq) for seq in Y])
    X = pad_sequence([Variable(torch.LongTensor(i)) for i in X])
    # X = X.reshape(X.shape[1], X.shape[0], X.shape[2])
    Y = pad_sequence([Variable(torch.LongTensor(i)) for i in Y], batch_first=True)
    print(X.shape, Y.shape)
    # exit(1)
    # model = Model(PL.N_STATES, PL.N_PHONEMES, 40, 40)
    # # print(model)
    # criterion = nn.CTCLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    #
    # for epoch in range(50):
    #     model.zero_grad()
    #     print(X.shape)
    #     print(X_lens.shape)
    #     out, out_lens = model(X, X_lens)
    #     loss = criterion(out, Y, out_lens, Y_lens)
    #     print('Epoch', epoch + 1, 'Loss', loss.item())
    #     loss.backward()
    #     optimizer.step()
