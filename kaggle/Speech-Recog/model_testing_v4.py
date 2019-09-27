import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable
from torch.utils import data
import csv
from speech_classification import Pred_Model

cuda = torch.cuda.is_available()
CONTEXT_SIZE = 12
FINAL_OUTPUT = []
OUTPUT_SIZE = 0


class MyDataset(data.Dataset):
    def __init__(self, X):
        self.X = X
        self.padX = X
        for i in range(len(self.padX)):
            self.padX[i] = np.pad(self.padX[i], ((CONTEXT_SIZE, CONTEXT_SIZE), (0, 0)), 'constant', constant_values=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        framex = self.padX[index].astype(float)  # flatten the input
        return framex


class SquaredDataset(Dataset):
    def __init__(self, mydata):
        super().__init__()
        num = 0
        finalx = mydata.X
        self.finaldict = {}
        for i in range(mydata.__len__()):
            x = mydata.__getitem__(i)
            for j in range(len(x) - 2 * CONTEXT_SIZE):
                self.finaldict[num] = (i, j + 2 * CONTEXT_SIZE + 1)
                num += 1
        self._x = finalx

    def __len__(self):
        return OUTPUT_SIZE

    def __getitem__(self, index):
        idx1, idx2 = self.finaldict[index][0], self.finaldict[index][1]
        x_item = (self._x[idx1][(idx2 - 2 * CONTEXT_SIZE - 1):idx2]).reshape(-1)
        return x_item


class Evaler():
    """
    A simple training cradle
    """

    def __init__(self, model, load_path=None):
        self.model = model
        if load_path is not None:
            self.model = torch.load(load_path)
        # self.optimizer = optimizer

    def eval_process(self, cur_loader):
        self.model.eval()
        self.model.to(device)
        for batch_idx, x in enumerate(cur_loader):
            X = Variable(x.float()).to(device)
            # Y = Variable(y).to(device)
            outputs = model(X)
            pred = outputs.data.max(1, keepdim=True)[1]
            FINAL_OUTPUT.extend(pred)


# def inference(model, loader, n_members):
#     correct = 0
#     for data, label in loader:
#         X = Variable(data)
#         Y = Variable(label)
#         out = model(X)
#         pred = out.data.max(1, keepdim=True)[1]
#         predicted = pred.eq(Y.data.view_as(pred))
#         correct += predicted.sum()
#     return correct.numpy() / n_members


if __name__ == "__main__":
    print("Cuda:{}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")
    testx = np.load("source_data.nosync/test.npy", allow_pickle=True)
    # testy = np.load("source_data.nosync/dev_labels.npy", allow_pickle=True)
    for i in testx:
        OUTPUT_SIZE += len(i)
    rawdata = MyDataset(X=testx)
    mydata = SquaredDataset(rawdata)
    model = Pred_Model()
    model.load_state_dict(torch.load('ep10new_model.pt', map_location=device))
    eval = Evaler(model)
    train_loader_args = dict(shuffle=False, batch_size=512, num_workers=0, pin_memory=True) if cuda \
        else dict(shuffle=False, batch_size=64)
    train_loader = data.DataLoader(mydata, **train_loader_args)
    eval.eval_process(train_loader)
print(len(FINAL_OUTPUT))
with open('ep10new_model.csv', mode='w') as csv_file:
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(FINAL_OUTPUT)):
        writer.writerow({'id': i, 'label': int(FINAL_OUTPUT[i])})
