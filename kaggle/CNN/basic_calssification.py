import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset

import cnn_params as P
import model_basic as M
import resnet as R

device = M.device

def get_output(model, test_loader):
    model.eval()
    output = []
    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        print(feats.shape)
        feature, outputs = model(feats)
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        output.extend(pred_labels)
    return output


if __name__ == "__main__":
    val_dataset = torchvision.datasets.ImageFolder(root='data.nosync/validation_classification/medium', transform=torchvision.transforms.ToTensor())
    # test_dataset = torchvision.datasets.ImageFolder(root='data.nosync/test_classification', transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.ImageFolder(root='data.nosync/validation_classification/medium', transform=torchvision.transforms.ToTensor())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)
    # print(val_dataset.classes)
    val_labels = [i[1] for i in val_dataset.samples]
    # label_ref = [(i, j) for i, j in enumerate(val_dataset.classes)]
    label_ref = [j for j in val_dataset.classes]
    s1 = len("data.nosync/test_classification/medium/")
    id_list = [i[0][s1:-4] for i in test_dataset.samples]
    ##########################################################
    model = M.network
    model.load_state_dict(torch.load('saved_models/basics/5437.pt', map_location=M.device))
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
