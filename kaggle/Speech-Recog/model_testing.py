import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from collections import namedtuple
from IPython.display import Image
import csv


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


### TEST ###
if __name__ == "__main__":
    testx = np.load("source_data.nosync/test.npy", allow_pickle=True)
    # trainx = np.load("train.npy", allow_pickle=True)
    model = Pred_Model()
    model.load_state_dict(torch.load('./saved_model.pt',map_location='cpu'))
    # test_acc = inference(model, test_loader, test_size)
    # print("Test accuracy of model optimizer with SGD: {0:.2f}".format(test_acc * 100))
    n = 0
    result = []
    for X in testx:
        x = Variable(torch.from_numpy(X).float())
        out = model(x)
        pred = out.data.max(1, keepdim=True)[1]
        n += len(pred)
        for i in pred:
            result.append(i)
    print(len(result))
    with open('output.csv', mode='w') as csv_file:
        fieldnames = ['id', 'label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(result)):
            writer.writerow({'id': i, 'label': int(result[i][0])})
