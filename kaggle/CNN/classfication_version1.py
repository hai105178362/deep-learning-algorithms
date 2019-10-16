import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import sys
import torch
import torchvision
import cnnmodel as M
import cnn_params as P
import torch.nn.functional as F

device = M.device


# def parse_data(datadir):
#     img_list = []
#     ID_list = []
#     for root, directories, filenames in os.walk(datadir):
#         for filename in filenames:
#             if filename.endswith('.jpg'):
#                 filei = os.path.join(root, filename)
#                 img_list.append(filei)
#                 ID_list.append(root.split('/')[-1])
#
#     # construct a dictionary, where key and value correspond to ID and target
#     uniqueID_list = list(set(ID_list))
#     class_n = len(uniqueID_list)
#     target_dict = dict(zip(uniqueID_list, range(class_n)))
#     label_list = [target_dict[ID_key] for ID_key in ID_list]
#
#     print('{}\t\t{}\n{}\t\t{}'.format('#Images', '#Labels', len(img_list), len(set(label_list))))
#     return img_list, label_list, class_n

def get_output(model, test_loader):
    model.eval()
    output = []
    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        feature, outputs = model(feats)
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        output.extend(pred_labels)
    return output


if __name__ == "__main__":
    val_dataset = torchvision.datasets.ImageFolder(root='data.nosync/validation_classification/medium', transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.ImageFolder(root='data.nosync/test_classification', transform=torchvision.transforms.ToTensor())
    # test_dataset = torchvision.datasets.ImageFolder(root='data.nosync/validation_classification/medium', transform=torchvision.transforms.ToTensor())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)
    # print(val_dataset.classes)
    val_labels = [i[1] for i in val_dataset.samples]
    # label_ref = [(i, j) for i, j in enumerate(val_dataset.classes)]
    label_ref = [j for j in val_dataset.classes]

    s1 = len("data.nosync/test_classification/medium/")
    id_list = [i[0][s1:-4] for i in test_dataset.samples]
    model = M.Resnet(P.num_feats, P.hidden_sizes, P.num_classes, P.feat_dim)
    model.load_state_dict(torch.load('saved_models/16-16-3-e29.pt', map_location=M.device))
    output = get_output(model, test_dataloader)
    result = [label_ref[output[i].item()] for i in range(len(output))]
    print(result)
    # print(val_labels)
    # print ( result == val_labels)
    # print(result[0].type('torch.DoubleTensor'))
    # print([label_ref[i][1] for i in result])
    # print(label_ref)
    with open("hw2p2_classification_sub.csv", 'w+') as f:
        f.write('id,label\n')
        for i, j in zip(id_list, result):
            f.write(str(i) + ',' + str(j) + '\n')
