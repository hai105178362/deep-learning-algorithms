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

cuda = torch.cuda.is_available()
CONTEXT_SIZE = 0


class MyDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.padX = X
        for i in range(len(self.padX)):
            self.padX[i] = np.pad(self.padX[i], ((CONTEXT_SIZE, CONTEXT_SIZE), (0, 0)), 'constant', constant_values=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        framex = self.padX[index].astype(float)  # flatten the input
        # X = self.X[index].astype(float).reshape(-1)  # flatten the input
        # Y = self.Y[index].astype(float)
        # framey = self.Y[index]
        return framex


class SquaredDataset(Dataset):
    def __init__(self, x):
        super().__init__()
        # print(x.shape)
        # print(x[0:14])
        # assert len(x) - 2 * CONTEXT_SIZE == len(y)
        newx = np.zeros((len(x) - 2 * CONTEXT_SIZE, 40 * (2 * CONTEXT_SIZE + 1)))
        # print(newx.shape)
        for i in range(CONTEXT_SIZE, len(newx) - 2 * CONTEXT_SIZE):
            # print(newx[i - CONTEXT_SIZE],x[i - CONTEXT_SIZE:(i + CONTEXT_SIZE + 1)].reshape(-1))
            newx[i - CONTEXT_SIZE] = x[i - CONTEXT_SIZE:(i + CONTEXT_SIZE + 1)].reshape(-1)
        self._x = newx
        # self._y = y
        # print(self._x[0], len(self._x))
        # print(self._y[0], len(self._y))
        # sys.exit(1)

    def __len__(self):
        return len(self._x)

    def __getitem__(self, index):
        x_item = self._x[index]
        # x_item = torch.cat([x_item, x_item.pow(2)])
        return x_item


class Pred_Model(nn.Module):

    def __init__(self):
        super(Pred_Model, self).__init__()
        self.fc1 = nn.Linear(40, 256)
        self.bnorm1 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, 128)
        self.bnorm2 = nn.BatchNorm1d(128)
        self.dp2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(128, 138)
        # self.bnorm3 = nn.BatchNorm1d(64)
        # self.dp3 = nn.Dropout(p=0.2)
        # self.fc4 = nn.Linear(64, 64)
        # self.bnorm4 = nn.BatchNorm1d(64)
        # self.dp4 = nn.Dropout(p=0.1)
        # self.fc5 = nn.Linear(64, 138)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.dp1(self.bnorm1(x))
        x = F.sigmoid(self.fc2(x))
        # x = self.dp2(self.bnorm2(x))
        x = F.log_softmax(self.fc3(x))
        # x = self.dp3(self.bnorm3(x))
        # x = F.relu(self.fc3(x))
        # x = self.dp4(self.bnorm4(x))
        # x = F.log_softmax(self.fc3(x))
        return x


def init_xavier(m):
    if type(m) == nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)



class evaler():
    """
    A simple training cradle
    """

    def __init__(self, model, load_path=None):
        self.model = model
        if load_path is not None:
            self.model = torch.load(load_path)
        # self.optimizer = optimizer

    def get_eval_info(self, cur_loader, criterion):
        # self.model.eval()
        # self.model.to(device)
        running_loss = 0.0
        # scheduler.step()
        correct = 0
        samples = 0
        start_time = time.time()
        arr = []
        refarr = []
        for batch_idx, x, in enumerate(cur_loader):
            # print(x,x.shape)
            # sys.exit(1)
            # self.optimizer.zero_grad()
            X = Variable(x.float()).to(device)
            # Y = Variable(y).to(device)
            outputs = model(X)
            pred = outputs.data.max(1, keepdim=True)[1]
            # predicted = pred.eq(Y.data.view_as(pred))
            # print(list(pred.view(1,-1)[0]))
            # print(Y.data.view(1,-1))
            predarr = (pred.numpy()[:, ::-1].reshape(-1))
            # ref = y.numpy()
            # correct += predicted.sum()
            # samples += len(y)
            # loss = criterion(outputs, Y)
            # running_loss += loss.item()
            # loss.backward()
            # self.optimizer.step()
            # arr.extend(predarr)
            # print(len(predarr))
            arr.extend(predarr)
        # print(arr)
        # print(refarr)
        # print(np.sum(np.array([arr]) == np.array([refarr])), "out of ", len(arr))
        # running_loss /= len(cur_loader)
        # print('Training Loss: ', running_loss, 'Time: ', end_time - start_time, 's')
        return arr
        # return int(correct), samples, arr, refarr


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
    testx = np.load("source_data.nosync/dev.npy", allow_pickle=True)
    # testy = np.load("source_data.nosync/dev_labels.npy", allow_pickle=True)
    # testx = np.load("dev_labels.npy", allow_pickle=True)
    #     trainx = np.load("dev.npy", allow_pickle=True)
    # trainy = np.load("train_labels.npy", allow_pickle=True)
    # trainx = np.load("train.npy", allow_pickle=True)
    # trainy = np.load("/content/drive/My Drive/Colab Notebooks/dev_labels.npy", allow_pickle=True)
    # trainx = np.load("/content/drive/My Drive/Colab Notebooks/dev.npy", allow_pickle=True)
    mydata = MyDataset(X=testx, Y=[0])
    model = Pred_Model()
    model.load_state_dict(torch.load('saved_model.pt', map_location=device))
    evaluation = evaler(model)
    # start_time = time.time()
    finalarr = []
    yarr = []
    tot_correct = 0
    tot_samples = 0
    print(mydata.__len__())
    for i in range(mydata.__len__()):
        # curx, cury = mydata.__getitem__(i)
        curx = mydata.__getitem__(i)
        # print(curx.shape,cury)
        test_dataset = SquaredDataset(curx)
        test_loader_args = dict(shuffle=False, batch_size=16, num_workers=0, pin_memory=True) if cuda \
            else dict(shuffle=True, batch_size=64)
        test_loader = data.DataLoader(test_dataset, **test_loader_args)
        # correct, samplesm, tmparr,refarr = trainer.get_eval_info(train_loader, criterion=nn.CrossEntropyLoss())
        refarr = evaluation.get_eval_info(test_loader, criterion=nn.CrossEntropyLoss())
        finalarr += refarr
        # yarr +=refarr
        # tot_samples += samplesm
        # tot_correct += correct
    # print(np.sum(np.array([finalarr]) == np.array([yarr])), "out of ", len(yarr))
    # print(len(testy))
    print("Finished, len = {} {}".format(len(finalarr), len(testx)))
    with open('tmpresult_2.csv', mode='w') as csv_file:
        fieldnames = ['id', 'label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(finalarr)):
            writer.writerow({'id': i, 'label': int(finalarr[i])})