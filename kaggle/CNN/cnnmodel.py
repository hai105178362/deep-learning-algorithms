import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import cnn_params as P
import tracewritter as wrt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BasicBlock(nn.Module):

    def __init__(self, channel_size, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_size)
        self.conv2 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel_size)
        self.conv3 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channel_size)
        # self.shortcut = nn.Conv2d(channel_size, channel_size, kernel_size=1, stride=1, bias=False)
        self.shortcut = nn.Sequential(
            nn.Conv2d(channel_size, channel_size, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channel_size))

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)),inplace=True)
        out = F.relu6(self.bn1(self.conv1(x)), inplace=True)
        # out = self.bn1(self.conv1(out))

        # out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn2(self.conv2(out))

        # out = F.relu(self.bn3(self.conv3(out)))
        # out = self.bn3(self.conv3(out))

        out += self.shortcut(x)
        out = F.relu6(out)
        return out


class Resnet(nn.Module):
    def __init__(self, num_feats, hidden_sizes, num_classes, feat_dim=10):
        super(Resnet, self).__init__()

        self.hidden_sizes = [num_feats] + hidden_sizes + [num_classes]

        self.layers = []
        for idx, channel_size in enumerate(hidden_sizes):
            self.layers.append(nn.Conv2d(in_channels=self.hidden_sizes[idx],
                                         out_channels=self.hidden_sizes[idx + 1],
                                         kernel_size=3, stride=2, bias=False))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(BasicBlock(channel_size=channel_size))
            self.layers.append(nn.BatchNorm2d(channel_size))
            self.layers.append(BasicBlock(channel_size=channel_size))
            # self.layers.append(nn.BatchNorm2d(channel_size))
            # self.layers.append(BasicBlock(channel_size=channel_size))

        self.layers = nn.Sequential(*self.layers)
        self.linear_label = nn.Linear(self.hidden_sizes[-2], self.hidden_sizes[-1], bias=False)

        # For creating the embedding to be passed into the Center Loss criterion
        self.linear_closs = nn.Linear(self.hidden_sizes[-2], feat_dim, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)

    def forward(self, x, evalMode=False):
        output = x
        output = self.layers(output)

        output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
        output = output.reshape(output.shape[0], output.shape[1])

        label_output = self.linear_label(output)
        label_output = label_output / torch.norm(self.linear_label.weight, dim=1)

        # Create the feature embedding for the Center Loss
        closs_output = self.linear_closs(output)
        closs_output = self.relu_closs(closs_output)

        return closs_output, label_output


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)


def train_closs(model, data_loader, test_loader, task='Classification', prev_acc=0):
    model.train()

    for epoch in range(P.numEpochs):
        start_time = time.time()
        avg_loss = 0.0
        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)

            optimizer_label.zero_grad()
            optimizer_closs.zero_grad()

            feature, outputs = model(feats)

            l_loss = criterion_label(outputs, labels.long())
            c_loss = criterion_closs(feature, labels.long())
            loss = l_loss + P.closs_weight * c_loss

            loss.backward()

            optimizer_label.step()
            # by doing so, weight_cent would not impact on the learning of centers
            for param in criterion_closs.parameters():
                param.grad.data *= (1. / P.closs_weight)
            optimizer_closs.step()

            avg_loss += loss.item()

            if batch_num % 500 == 499:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch + 1, batch_num + 1, avg_loss / 50))
                avg_loss = 0.0

            torch.cuda.empty_cache()
            del feats
            del labels
            del loss

        if task == 'Classification':
            val_loss, val_acc = test_classify_closs(model, test_loader)
            train_loss, train_acc = test_classify_closs(model, data_loader)
            end_time = time.time()
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(train_loss, train_acc, val_loss, val_acc))
            print("Time: {}".format(end_time-start_time))

            if train_acc >= 0.45 or val_acc >= 0.45 or (epoch+1 >= 10 and train_acc + val_acc > prev_acc):
                d = datetime.datetime.today()
                record = "{}-{}-{}-e{}".format(d.day, d.hour, d.minute, epoch+1)
                modelpath = "saved_models/{}.pt".format(record)
                torch.save(model.state_dict(), modelpath)
                print("Model saved at: {}".format(modelpath))
                wrt.recordtrace(train_acc, train_loss, val_acc, val_loss, epoch+1)
                prev_acc = train_acc + val_acc
        # else:
        #     test_verify(model, test_loader)


def test_classify_closs(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        feature, outputs = model(feats)

        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)

        l_loss = criterion_label(outputs, labels.long())
        c_loss = criterion_closs(feature, labels.long())
        loss = l_loss + P.closs_weight * c_loss

        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()] * feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy / total


class CenterLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes, feat_dim, device=torch.device('cpu')):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss


def test_classify_closs(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0
    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        feature, outputs = model(feats)

        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)

        l_loss = criterion_label(outputs, labels.long())
        c_loss = criterion_closs(feature, labels.long())
        loss = l_loss + P.closs_weight * c_loss

        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()] * feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy / total


network = Resnet(P.num_feats, P.hidden_sizes, P.num_classes)
criterion_label = nn.CrossEntropyLoss()
criterion_closs = CenterLoss(P.num_classes, P.feat_dim, P.device)
optimizer_label = torch.optim.SGD(network.parameters(), lr=P.learningRate, weight_decay=P.weightDecay, momentum=0.9)
# optimizer_label = torch.optim.Adam(network.parameters(), lr=P.learningRate)
optimizer_closs = torch.optim.SGD(criterion_closs.parameters(), lr=P.lr_cent)
# optimizer_closs = torch.optim.Adam(criterion_closs.parameters(), lr=P.lr_cent)
