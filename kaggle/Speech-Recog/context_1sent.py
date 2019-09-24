'''
Loss: 2.815763552983602   Correct: 2974025  Samples: 15388610 Accuracy:0.0  Time: 265.41697573661804
epoch:2
Loss: 2.784127950668335   Correct: 3289823  Samples: 15388610 Accuracy:0.0  Time: 262.2106890678406
epoch:3
Loss: 2.800300041834513   Correct: 3369028  Samples: 15388610 Accuracy:0.0  Time: 260.2379970550537
epoch:4
Loss: 2.7539426485697427   Correct: 3352017  Samples: 15388610 Accuracy:0.0  Time: 261.23384952545166
epoch:5
Loss: 2.6530216534932456   Correct: 3366201  Samples: 15388610 Accuracy:0.0  Time: 262.9407832622528
epoch:6
Loss: 2.628347476323446   Correct: 3402009  Samples: 15388610 Accuracy:0.0  Time: 259.39038920402527
epoch:7
Loss: 2.654765764872233   Correct: 3414673  Samples: 15388610 Accuracy:0.0  Time: 260.15447330474854
epoch:8
Loss: 2.6403125127156577   Correct: 3385275  Samples: 15388610 Accuracy:0.0  Time: 264.572651386261
'''
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

cuda = torch.cuda.is_available()


class MyDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        X = self.X[index].astype(float)  # flatten the input
        # X = self.X[index].astype(float).reshape(-1)  # flatten the input
        # Y = self.Y[index].astype(float)
        Y = self.Y[index]
        return X, Y


class SquaredDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        assert len(x) == len(y)
        self._x = x
        self._y = y

    def __len__(self):
        return len(self._x)

    def __getitem__(self, index):
        x_item = self._x[index]
        # x_item = torch.cat([x_item, x_item.pow(2)])
        return x_item, self._y[index]


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
        x = self.dp1(self.bnorm1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.dp2(self.bnorm2(x))
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


class Trainer():
    """
    A simple training cradle
    """

    def __init__(self, model, optimizer, load_path=None):
        self.model = model
        if load_path is not None:
            self.model = torch.load(load_path)
        self.optimizer = optimizer

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def train_per_epoch(self, x, y):
        model.train()
        model.to(device)
        running_loss = 0.0
        # scheduler.step()
        correct = 0
        samples = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            self.optimizer.zero_grad()
            X = Variable(x.float()).to(device)
            Y = Variable(y).to(device)
            if len(X) == 1:
                continue
            out = model(X)
            pred = out.data.max(1, keepdim=True)[1]
            predicted = pred.eq(Y.data.view_as(pred))
            correct += predicted.sum()
            samples += len(y)
            loss = F.nll_loss(out, Y)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            # if (epoch + 1) % 20 == 0:
        # print("loss:{} corret:{} @ {}".format(loss,correct,256*len(train_loader)))
        running_loss /= len(train_loader)
        return correct, samples, running_loss


def inference(model, loader, n_members):
    correct = 0
    for data, label in loader:
        X = Variable(data)
        Y = Variable(label)
        out = model(X)
        pred = out.data.max(1, keepdim=True)[1]
        predicted = pred.eq(Y.data.view_as(pred))
        correct += predicted.sum()
    return correct.numpy() / n_members


if __name__ == "__main__":
    print("Cuda:{}".format(cuda))
    # trainx = np.load("source_data.nosync/train.npy", allow_pickle=True)[:200]
    # trainy = np.load("source_data.nosync/train_labels.npy", allow_pickle=True)[:200]
    trainy = np.load("train_labels.npy", allow_pickle=True)
    trainx = np.load("train.npy", allow_pickle=True)
    mydata = MyDataset(X=trainx, Y=trainy)
    print("data loaded")
    device = torch.device("cuda" if cuda else "cpu")
    model = Pred_Model()
    model.apply(init_xavier)
    optimizer = optim.SGD(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    trainer = Trainer(model, optimizer)
    nepoch = 10
    tot_correct = 0
    tot_samples = 0
    for epoch in range(nepoch):
        print("epoch:{}".format(epoch + 1))
        tot_correct = 0
        tot_samples = 0
        start_time = time.time()
        for idx in range(len(mydata.X)):
            curx, cury = mydata.__getitem__(idx)
            if idx > 0:
                prevx, prevy = mydata.__getitem__(idx - 1)
            else:
                prevx, prevy = curx * 0, cury * 0
            if idx < len(mydata.X) - 1:
                nextx, nexty = mydata.__getitem__(idx + 1)
            else:
                nextx, nexty = curx * 0, cury * 0

            newx = np.concatenate([prevx, curx, nextx])
            newy = np.concatenate([prevy, cury, nexty])
            # print(prevx.shape, prevy.shape)
            # print(curx.shape, cury.shape)
            # print(nextx.shape,nexty.shape)
            # print(newx.shape,newy.shape)
            # print(len(curx),len(cury))
            # print(len(newx),len(newy))
            # sys.exit(1)
            train_dataset = SquaredDataset(curx, cury)
            train_loader_args = dict(shuffle=True, batch_size=256, num_workers=0, pin_memory=True) if cuda \
                else dict(shuffle=True, batch_size=256)
            train_loader = data.DataLoader(train_dataset, **train_loader_args)
            a, b, c = trainer.train_per_epoch(x=newx, y=newy)
            tot_correct += a
            tot_samples += b
            # for i in range(1, len(mydata.X)):
            #     for cur_x, cur_y in zip(mydata.X[i - 1:i + 1], mydata.Y[i - 1:i + 1]):
            #         trainer.train_per_epoch(nepochs=80, x=cur_x, y=cur_y)
        end_time = time.time()
        print("Loss: {}   Correct: {}  Samples: {} Accuracy:{}  Time: {}".format(c, tot_correct, tot_samples, float(tot_correct / tot_samples), end_time - start_time))
    trainer.save_model('./saved_model.pt')
    print("Model Saved! Good Luck! :D")
