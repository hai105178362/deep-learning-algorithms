import torch
import torchvision
from torch.utils.data import Dataset


feat_dim = 10
closs_weight = 0.6
lr_cent = 0.5
numEpochs = 100
num_feats = 3
learningRate = 0.1
weightDecay = 1e-5

######Parameters fo basic model######
# closs_weight = 0.6
# hidden_sizes = [32, 64, 96, 144]
# batch_size = 128
#####################################

######Parameters fo Resnet#####
closs_weight = 0.6
# layers = [3,4,6,3]
layers = [3,4,6,3]
# hidden_sizes = [32, 64, 128, 128]
hidden_sizes = [32, 64, 128, 128]
batch_size = 128
###############################


allspec = "closs_weight:{}\t".format(closs_weight) + "lr_cent:{}\t".format(lr_cent) + "feat_dim:{}\n".format(feat_dim) \
          + "num_feats:{}\t\t".format(num_feats) + "learningRate:{}\n".format(learningRate) + "weight_decay:{}\t".format(weightDecay) + \
          "hidden_size:{}\t\tbatch_size:{}".format(hidden_sizes, batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = torchvision.datasets.ImageFolder(root='data.nosync/validation_classification/medium', transform=torchvision.transforms.ToTensor())
# train_dataset = torchvision.datasets.ImageFolder(root='dataset/train_data/medium', transform=torchvision.transforms.ToTensor())
dev_dataset = torchvision.datasets.ImageFolder(root='data.nosync/validation_classification/medium', transform=torchvision.transforms.ToTensor())
# dev_dataset = torchvision.datasets.ImageFolder(root='dataset/validation_classification/medium', transform=torchvision.transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
num_classes = len(train_dataset.classes)


def modelpath(record):
    return "saved_models/{}.pt".format(record)
