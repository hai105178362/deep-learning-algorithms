import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys

NUM_EPOCHS = 10
# NUM_FEATS = 3
NUM_FEATS = 3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-5
# HIDDEN_SIZE = [32, 64]
HIDDEN_SIZE = [224, 224, 96, 64]
CLOSS_WEIGHT = 1
LR_CENT = 0.5
# feat_dim = 10
FEAT_DIM = 2300

class ImageDataset(Dataset):
    def __init__(self, file_list, target_list):
        self.file_list = file_list
        self.target_list = target_list
        self.n_class = len(list(set(target_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        label = self.target_list[index]
        return img, label


def parse_data(datadir):
    img_list = []
    ID_list = []
    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                img_list.append(filei)
                ID_list.append(root.split('/')[-1])

    # construct a dictionary, where key and value correspond to ID and target
    uniqueID_list = list(set(ID_list))
    class_n = len(uniqueID_list)
    target_dict = dict(zip(uniqueID_list, range(class_n)))
    label_list = [target_dict[ID_key] for ID_key in ID_list]

    print('{}\t\t{}\n{}\t\t{}'.format('#Images', '#Labels', len(img_list), len(set(label_list))))
    return img_list, label_list, class_n


class BasicBlock(nn.Module):

    def __init__(self, channel_size, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(channel_size)
        self.dropout1 = nn.Dropout2d(0.5)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(channel_size, channel_size, kernel_size=1, stride=stride, bias=False)
        # self.bn2 = nn.BatchNorm2d(channel_size)
        self.dropout2 = nn.Dropout2d(0.5)

    def forward(self, x):
        out = F.relu(self.dropout1(self.conv1(x)))
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout2(self.conv1(out))
        # out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Network(nn.Module):
    def __init__(self, num_feats, hidden_sizes, num_classes, feat_dim=FEAT_DIM):
        super(Network, self).__init__()

        self.hidden_sizes = [num_feats] + hidden_sizes + [num_classes]

        self.layers = []
        for idx, channel_size in enumerate(hidden_sizes):
            self.layers.append(nn.Conv2d(in_channels=self.hidden_sizes[idx],
                                         out_channels=self.hidden_sizes[idx + 1],
                                         kernel_size=3, stride=2, bias=False))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(BasicBlock(channel_size=channel_size))

        self.layers = nn.Sequential(*self.layers)
        self.linear_label = nn.Linear(self.hidden_sizes[-2], self.hidden_sizes[-1], bias=False)

        # For creating the embedding to be passed into the Center Loss criterion
        self.linear_closs = nn.Linear(self.hidden_sizes[-2], feat_dim, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)

    def forward(self, x, evalMode=False):
        output = x
        output = self.layers(output)

        # output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
        output = F.max_pool2d(output, [output.size(2), output.size(3)], stride=1)
        output = output.reshape(output.shape[0], output.shape[1])

        label_output = self.linear_label(output)
        label_output = label_output / torch.norm(self.linear_label.weight, dim=1)

        # Create the feature embedding for the Center Loss
        if evalMode == False:
            closs_output = self.linear_closs(output)
            closs_output = self.relu_closs(closs_output)
            return closs_output, label_output
        return label_output


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)
def test_verify(model, test_loader):
    raise NotImplementedError


############## CLOSS
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


def train_closs(model, data_loader, test_loader, task='Classification'):
    model.train()
    PATH = "saved_models/cnn_epoch{}.pt".format('test')
    torch.save(model.state_dict(), PATH)
    for epoch in range(NUM_EPOCHS):
        avg_loss = 0.0
        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)

            optimizer_label.zero_grad()
            optimizer_closs.zero_grad()

            feature, outputs = model(feats)

            l_loss = criterion_label(outputs, labels.long())
            c_loss = criterion_closs(feature, labels.long())
            loss = l_loss + CLOSS_WEIGHT * c_loss

            loss.backward()

            optimizer_label.step()
            # by doing so, weight_cent would not impact on the learning of centers
            for param in criterion_closs.parameters():
                param.grad.data *= (1. / CLOSS_WEIGHT)
            optimizer_closs.step()

            avg_loss += loss.item()

            if batch_num % 50 == 49:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch + 1, batch_num + 1, avg_loss / 50))
                avg_loss = 0.0

            torch.cuda.empty_cache()
            del feats
            del labels
            del loss

        if task == 'Classification':
            val_loss, val_acc = test_classify_closs(model, test_loader)
            train_loss, train_acc = test_classify_closs(model, data_loader)
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(train_loss, train_acc, val_loss, val_acc))
            PATH = "saved_models/cnn_epoch{}.pt".format(epoch)
            torch.save(model.state_dict(), PATH)
            # if train_acc >= 0.7 or val_acc >= 0.7:
            #     PATH = "saved_models/cnn_epoch{}.pt".format(epoch)
            #     torch.save(model.state_dict(), PATH)
            #     print("==========================================================================")
            #     print("Model Saved with train accuracy {:.5f} and val accuracy {:.5f} at epoch {}".format(train_acc, val_acc, epoch))
            #     print("==========================================================================")
        else:
            test_verify(model, test_loader)


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
        loss = l_loss + CLOSS_WEIGHT * c_loss

        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()] * feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy / total



# if __name__ == '__main__':
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
# TRAIN_PATH = 'data.nosync/11785-f19-hw2p2-classification/11-785hw2p2-f19/train_data/medium'
# TRAIN_PATH = 'devset/medium'
# TRAIN_PATH = 'data.nosync/11785-f19-hw2p2-classification/11-785hw2p2-f19/validation_classification/medium'
TRAIN_PATH = 'dataset/validation_classification/medium'

# VAL_PATH = 'data.nosync/11785-f19-hw2p2-classification/11-785hw2p2-f19/validation_classification/medium/'
# VAL_PATH = 'devset/medium_dev'
# VAL_PATH = 'data.nosync/11785-f19-hw2p2-classification/11-785hw2p2-f19/validation_classification/medium'
VAL_PATH = 'dataset/validation_classification/medium'

img_list, label_list, class_n = parse_data(TRAIN_PATH)
trainset = ImageDataset(img_list, label_list)
train_data_item, train_data_label = trainset.__getitem__(0)
print('data item shape: {}\t data item label: {}'.format(train_data_item.shape, train_data_label))
dataloader = DataLoader(trainset, batch_size=10, shuffle=True, num_workers=1, drop_last=False)
imageFolder_dataset = torchvision.datasets.ImageFolder(root=TRAIN_PATH,
                                                       transform=torchvision.transforms.ToTensor())
imageFolder_dataloader = DataLoader(imageFolder_dataset, batch_size=10, shuffle=True, num_workers=1)
print(imageFolder_dataset.__len__(), len(imageFolder_dataset.classes))

train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_PATH,
                                                 transform=torchvision.transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10,
                                               shuffle=True, num_workers=8)

dev_dataset = torchvision.datasets.ImageFolder(root=VAL_PATH,
                                               transform=torchvision.transforms.ToTensor())
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=10,
                                             shuffle=True, num_workers=8)

NUM_CLASSES = len(train_dataset.classes)


if __name__ == "__main__":
    network = Network(NUM_FEATS, HIDDEN_SIZE, NUM_CLASSES, FEAT_DIM)
    network.apply(init_weights)

    criterion_label = nn.CrossEntropyLoss()
    criterion_closs = CenterLoss(NUM_CLASSES, FEAT_DIM, device)
    optimizer_label = torch.optim.SGD(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=0.9)
    # optimizer_label = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    optimizer_closs = torch.optim.SGD(criterion_closs.parameters(), lr=LR_CENT)
    # optimizer_closs = torch.optim.Adam(criterion_closs.parameters(), lr=LR_CENT)

    network.train()
    network.to(device)
    train_closs(network, train_dataloader, dev_dataloader)
