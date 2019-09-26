import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import csv
import time

cuda = torch.cuda.is_available()
CONTEXT_SIZE = 14
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
        # framey = self.Y[index]
        return framex


class SquaredDataset(Dataset):
    def __init__(self, mydata):
        super().__init__()
        num = 0
        finalx = mydata.X
        finaly = []
        self.finaldict = {}
        for i in range(mydata.__len__()):
            x = mydata.__getitem__(i)
            # finaly.extend(y)
            # # finalx.append(x)
            for j in range(len(x) - 2 * CONTEXT_SIZE):
                self.finaldict[num] = (i, j + 2 * CONTEXT_SIZE + 1)
                num += 1
        self._x = finalx
        # self._y = finaly

    def __len__(self):
        return OUTPUT_SIZE

    def __getitem__(self, index):
        idx1, idx2 = self.finaldict[index][0], self.finaldict[index][1]
        x_item = (self._x[idx1][(idx2 - 2 * CONTEXT_SIZE - 1):idx2]).reshape(-1)
        # print(x_item)
        # sys.exit(1)
        # print(idx2 - 2 * CONTEXT_SIZE - 1, idx2)
        # sys.exit(1)
        # x_item = torch.cat([x_item, x_item.pow(2)])
        return x_item


class Pred_Model(nn.Module):

    def __init__(self):
        super(Pred_Model, self).__init__()
        self.fc1 = nn.Linear(40 * (1 + 2 * CONTEXT_SIZE), 1024)
        self.bnorm1 = nn.BatchNorm1d(1024)
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(1024, 1024)
        self.bnorm2 = nn.BatchNorm1d(1024)
        self.dp2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(1024, 512)
        self.bnorm3 = nn.BatchNorm1d(512)
        self.dp3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(512, 512)
        self.bnorm4 = nn.BatchNorm1d(512)
        self.dp4 = nn.Dropout(p=0.1)
        self.fc5 = nn.Linear(512, 256)
        self.bnorm5 = nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(256, 138)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if len(x) > 1:
            x = self.bnorm1(x)
        #             x = self.dp1(x)
        #             x = self.dp1(self.bnorm1(x))

        x = F.relu(self.fc2(x))
        if len(x) > 1:
            x = self.bnorm2(x)
        #             x = self.dp2(x)
        #             x = self.dp1(self.bnorm1(x))

        x = F.relu(self.fc3(x))
        if len(x) > 1:
            x = self.bnorm3(x)

        x = F.relu(self.fc4(x))
        # if len(x) > 1:
        x = self.bnorm4(x)

        x = F.sigmoid(self.fc5(x))
        # if len(x) > 1:
        x = self.bnorm5(x)

        # x = F.sigmoid(self.fc5)
        x = F.log_softmax(self.fc6(x))
        return x


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
    model.load_state_dict(torch.load('now_saved_model.pt', map_location=device))
    eval = Evaler(model)
    train_loader_args = dict(shuffle=False, batch_size=512, num_workers=0, pin_memory=True) if cuda \
        else dict(shuffle=False, batch_size=64)
    train_loader = data.DataLoader(mydata, **train_loader_args)
    eval.eval_process(train_loader)
print(len(FINAL_OUTPUT))
with open('tmpresult_test.csv', mode='w') as csv_file:
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(FINAL_OUTPUT)):
        writer.writerow({'id': i, 'label': int(FINAL_OUTPUT[i])})
