import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
cuda = torch.cuda.is_available()


class MyDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        X = self.X[index].float().reshape(-1)  # flatten the input
        Y = self.Y[index].long()
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

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dp1(self.bnorm1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.dp2(self.bnorm2(x))
        x = F.log_softmax(self.fc3(x))
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

    def run(self, nepochs, x, y):
        train_dataset = SquaredDataset(x, y)
        train_loader_args = dict(shuffle=True, batch_size=256, num_workers=0, pin_memory=True) if cuda \
            else dict(shuffle=True, batch_size=64)
        train_loader = data.DataLoader(train_dataset, **train_loader_args)
        for epoch in range(nepochs):
            # print("Epoch", epoch)
            model.train()
            model.to(device)
            running_loss = 0.0
            scheduler.step()
            correct = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                self.optimizer.zero_grad()
                X = Variable(x.float()).to(device)
                Y = Variable(y).to(device)
                out = model(X)
                pred = out.data.max(1, keepdim=True)[1]
                predicted = pred.eq(Y.data.view_as(pred))
                correct += predicted.sum()
                loss = F.nll_loss(out, Y)
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            if (epoch + 1) % 20 == 0:
                print("epoch:{}   loss:{} correct:{} out of {}".format(epoch + 1, loss, correct, 64*len(train_loader)))


def inference(model, loader, n_members):
    correct = 0
    for data, label in loader:
        X = Variable(data.view(-1, 40))
        Y = Variable(label)
        out = model(X)
        pred = out.data.max(1, keepdim=True)[1]
        predicted = pred.eq(Y.data.view_as(pred))
        correct += predicted.sum()
    return correct.numpy() / n_members


if __name__ == "__main__":
    trainy = np.load("train_labels.npy", allow_pickle=True)[:200]
    trainx = np.load("train.npy", allow_pickle=True)[:200]
    mydata = MyDataset(X=trainx, Y=trainy)
    device = torch.device("cuda" if cuda else "cpu")
    model = Pred_Model()
    model.apply(init_xavier)
    optimizer = optim.SGD(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    trainer = Trainer(model, optimizer)
    for i in range(1, len(mydata.X)):
        for cur_x, cur_y in zip(mydata.X[i - 1:i + 1], mydata.Y[i - 1:i + 1]):
            trainer.run(nepochs=80, x=cur_x, y=cur_y)
    trainer.save_model('./saved_model.pt')
    print("testing")

    ### TEST ###
    testmodel = Pred_Model()
    testmodel.load_state_dict(torch.load('./saved_model.pt'))
    testx = (np.load("test.npy", allow_pickle=True))[:5]
    # testy = (np.load("test.npy", allow_pickle=True))[:5]
    output = testmodel(Variable(testx))
