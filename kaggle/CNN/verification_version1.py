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
        feature, outputs = model(feats)
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        output.extend(pred_labels)
    return output


if __name__ == "__main__":
    val_dataset = torchvision.datasets.ImageFolder(root='data.nosync/validation_verification', transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.ImageFolder(root='data.nosync/test_verification', transform=torchvision.transforms.ToTensor())
    # test_dataset = torchvision.datasets.ImageFolder(root='data.nosync/validation_classification/medium', transform=torchvision.transforms.ToTensor())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)
    print(test_dataset.samples)
