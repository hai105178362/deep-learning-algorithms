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
from main import Network, NUM_FEATS, HIDDEN_SIZE, NUM_CLASSES
import csv


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
    return img_list, label_list, class_n, ID_list


def get_result(model, test_loader):
    model.eval()
    accuracy = 0
    total = 0
    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        feature, outputs = model(feats)
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        del feats
        del labels

    # model.train()
    return pred_labels


if __name__ == "__main__":
    print("Testing Procedure")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device is: ", device)
    model = Network(num_feats=NUM_FEATS, num_classes=NUM_CLASSES, hidden_sizes=HIDDEN_SIZE)
    model.load_state_dict(torch.load('saved_models/cnn_epoch6.pt', map_location=device))
    REF_PATH = 'data.nosync/11785-f19-hw2p2-classification/11-785hw2p2-f19/validation_classification/medium'
    dev_dataset = torchvision.datasets.ImageFolder(root=REF_PATH,
                                                   transform=torchvision.transforms.ToTensor())
    img_list, label_list, class_n, ID_list = parse_data(REF_PATH)
    ref_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=10,
                                                 shuffle=False, num_workers=8)
    final_result = get_result(model, ref_dataloader)
    print("Predict: ", final_result)
    print("Ref:", label_list)
    print("ID:",ID_list)
    #
    reflabel = []
    for batch_num, (feats, labels) in enumerate(ref_dataloader):
        feats, labels = feats.to(device), labels.to(device)
        reflabel.append(labels)
    print("Ref from dataloader: ", reflabel)
    # with open('result.csv', mode='w') as csv_file:
    #     fieldnames = ['id', 'label']
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for i in range(len(FINAL_OUTPUT)):
    #         writer.writerow({'id': i, 'label': int(FINAL_OUTPUT[i])})
