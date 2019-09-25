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
import csv
# from speech_classification import Pred_Model

cuda = torch.cuda.is_available()
CONTEXT_SIZE = 12
device = torch.device("cuda" if cuda else "cpu")


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
    def __init__(self, x):
        super().__init__()
        newx = np.zeros((len(x) - 2 * CONTEXT_SIZE, 40 * (2 * CONTEXT_SIZE + 1)))
        for i in range(CONTEXT_SIZE, len(newx) - 2 * CONTEXT_SIZE):
            newx[i - CONTEXT_SIZE] = x[i - CONTEXT_SIZE:(i + CONTEXT_SIZE + 1)].reshape(-1)
        self._x = newx

    def __len__(self):
        return len(self._x)

    def __getitem__(self, index):
        x_item = self._x[index]
        return x_item


class Pred_Model(nn.Module):

    def __init__(self):
        super(Pred_Model, self).__init__()
        self.fc1 = nn.Linear(40 * (1 + 2 * CONTEXT_SIZE), 1024)
        self.bnorm1 = nn.BatchNorm1d(1024)
        self.dp1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(1024, 512)
        self.bnorm2 = nn.BatchNorm1d(512)
        self.dp2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(512, 512)
        self.bnorm3 = nn.BatchNorm1d(512)
        # self.dp3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(512, 512)
        self.bnorm4 = nn.BatchNorm1d(512)
        # self.dp4 = nn.Dropout(p=0.1)
        self.fc5 = nn.Linear(512, 256)
        self.bnorm5 = nn.BatchNorm1d(256)

        self.fc6 = nn.Linear(256, 138)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if len(x) > 1:
            x = self.dp1(self.bnorm1(x))

        x = F.relu(self.fc2(x))
        if len(x) > 1:
            x = self.dp2(self.bnorm2(x))

        x = F.sigmoid(self.fc3(x))
        if len(x) > 1:
            x = self.bnorm3(x)

        x = F.sigmoid(self.fc4(x))
        if len(x) > 1:
            x = self.bnorm4(x)

        x = F.sigmoid(self.fc5(x))
        if len(x) > 1:
            x = self.bnorm5(x)

        # x = F.sigmoid(self.fc5)
        x = F.log_softmax(self.fc6(x))
        return x


def init_xavier(m):
    if type(m) == nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)


def getoutput(model, cur_loader):
    # model.to(device)
    predarr = []
    for batch_idx, x, in enumerate(cur_loader):
        X = Variable(x.float()).to(device)
        outputs = model(X)
        pred = outputs.data.max(1, keepdim=True)[1]
        # print(pred)
        for i in pred:
            predarr.append(i)
    return predarr


### TEST ###
if __name__ == "__main__":
    testx = np.load("source_data.nosync/test.npy", allow_pickle=True)
    # print(testx)
    model = Pred_Model()
    model.load_state_dict(torch.load('saved_model.pt', map_location='cpu'))
    mydata = MyDataset(X=testx)
    result = []
    for i in range(mydata.__len__()):
        curx = mydata.__getitem__(i)
        # print(curx.shape,cury)
        test_dataset = SquaredDataset(curx)
        test_loader_args = dict(shuffle=True, batch_size=512, num_workers=0, pin_memory=True) if cuda \
            else dict(shuffle=True, batch_size=256)
        test_loader = data.DataLoader(test_dataset, **test_loader_args)
        tmp = getoutput(model, test_loader)
        for i in tmp:
            result.append(i)
    print(len(result))
    # print(result)
    # print(out)
    # print(len(result))
    with open('context_12.csv', mode='w') as csv_file:
        fieldnames = ['id', 'label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(result)):
            writer.writerow({'id': i, 'label': int(result[i][0])})
