import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import cnn_params as P
import tracewritter as wrt
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_channel, stride=1):
    return nn.Conv2d(in_planes, out_channel, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, base_width=64, inflate=1):
        super(Bottleneck, self).__init__()
        bn = nn.BatchNorm2d
        width = int(out_channel * (base_width / 64.))
        self.conv1 = conv1x1(in_channel, width)
        self.bn1 = bn(width)
        # self.dp1 = nn.Dropout(p=0.2)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = bn(width)
        self.conv3 = conv1x1(width, out_channel * self.expansion)
        self.bn3 = bn(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = F.relu6(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.dp1(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2300, width=64, inflate=None):
        super(ResNet, self).__init__()
        bn = nn.BatchNorm2d
        self._bn = nn.BatchNorm2d
        self.in_channel = 64
        self.inflation = 1
        if inflate is None:
            inflate = [False, False, False]
        self.base_width = width
        self.conv1 = nn.Conv2d(
            3, self.in_channel, kernel_size=3, stride=2, padding=3, bias=False)
        # self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=3, stride=2, padding=3, bias=False)
        self.bn1 = bn(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.add_layer(block, P.hidden_sizes[0], layers[0])
        self.layer2 = self.add_layer(block, P.hidden_sizes[1], layers[1], stride=2, inflate=inflate[0])
        self.layer3 = self.add_layer(block, P.hidden_sizes[2], layers[2], stride=2, inflate=inflate[1])
        self.layer4 = self.add_layer(block, P.hidden_sizes[3], layers[3], stride=2, inflate=inflate[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            init_weights(m)
        self.linear_label = nn.Linear(P.hidden_sizes[-1] * block.expansion, P.num_classes, bias=False)
        self.linear_closs = nn.Linear(P.hidden_sizes[-1] * block.expansion, P.feat_dim, bias=False)
        self.relu_closs = nn.ReLU6(inplace=True)

    def add_layer(self, block, out_channel, blocks, stride=1, inflate=False):
        bn = self._bn
        downsample = None
        prev_inf = self.inflation
        if inflate:
            self.inflation *= stride
            stride = 1
        if stride != 1 or self.in_channel != out_channel * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channel, out_channel *
                        block.expansion, stride),
                bn(out_channel * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample,
                            self.base_width, prev_inf))
        self.in_channel = out_channel * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channel, out_channel,
                                base_width=self.base_width, inflate=self.inflation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        output = F.avg_pool2d(x, [x.size(2), x.size(3)], stride=1)
        output = output.reshape(output.shape[0], output.shape[1])
        # print(output.shape)
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
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch +
                                                                      1, batch_num + 1, avg_loss / 50))
                avg_loss = 0.0

            torch.cuda.empty_cache()
            del feats
            del labels
            del loss

        if task == 'Classification':
            val_loss, val_acc = test_classify_closs(model, test_loader)
            train_loss, train_acc = test_classify_closs(model, data_loader)
            end_time = time.time()
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}\tTime: {}'.
                  format(train_loss, train_acc, val_loss, val_acc, end_time - start_time))
            wrt.recordtrace(train_acc, train_loss,
                            val_acc, val_loss, epoch + 1)
            if train_acc >= 0.55 or val_acc >= 0.55 or (epoch + 1 >= 10 and train_acc + val_acc > prev_acc):
                d = datetime.datetime.today()
                record = "{}-{}-{}-e{}".format(d.day,
                                               d.hour, d.minute, epoch + 1)
                modelpath = "saved_models/{}.pt".format(record)
                torch.save(model.state_dict(), modelpath)
                print("Model saved at: {}".format(modelpath))
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

        self.centers = nn.Parameter(torch.randn(
            self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
            torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(
                self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            # for numerical stability
            value = value.clamp(min=1e-12, max=1e+12)
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


network = ResNet(Bottleneck, P.layers)
criterion_label = nn.CrossEntropyLoss()
criterion_closs = CenterLoss(P.num_classes, P.feat_dim, P.device)
# optimizer_label = torch.optim.SGD(network.parameters(), lr=P.learningRate, weight_decay=P.weightDecay, momentum=0.9)
optimizer_label = torch.optim.Adam(
    network.parameters(), lr=P.learningRate, weight_decay=P.weightDecay)
# optimizer_closs = torch.optim.SGD(criterion_closs.parameters(), lr=P.lr_cent)
optimizer_closs = torch.optim.Adam(criterion_closs.parameters(), lr=P.lr_cent)
