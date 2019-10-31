from torch.nn import CTCLoss
import torch
import numpy as np
import sys
import helper.phoneme_list as phol

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
    # task = sys.argv[1]
    task = "dev"
    if task == "train":
        xpath = trainxpath
        ypath = trainypath
    else:
        xpath = devxpath
        ypath = devypath
    x, y = load_data(xpath, ypath)
    print(x.shape, y.shape)
    word = ""
    for i in y[0]:
        word += phol.PHONEME_MAP[i]
    print(word)

    # print(len(phol.PHONEME_LIST))
