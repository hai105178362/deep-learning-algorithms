import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
import cnn_params as P
import model_basic as B
import resnet as R
from scipy.spatial.distance import cdist
import torch.nn as nn
from scipy import spatial

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_output(model, test_loader, idx):
    model.eval()
    output = []
    for batch_num, (feats, labels) in enumerate(test_loader):
        idx += 1
        feats, labels = feats.to(device), labels.to(device)
        feature, outputs = model(feats)
        # print(feature)
        # print(outputs)
        # _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        # print(pred_labels)
        # pred_labels = pred_labels.view(-1)
        output.extend(outputs)
        if batch_num>=10:
            return output
    return outputs


if __name__ == "__main__":
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    model_option = input("Model type? (b/r): ")
    preflex = "data.nosync/"
    if model_option == 'b':
        namearr = []
        final = []
        idx = 0
        model = B.network
        model.load_state_dict(torch.load('saved_models/basics/5437.pt', map_location=device))
        val_dataset = torchvision.datasets.ImageFolder(root=preflex + 'validation_verification', transform=torchvision.transforms.ToTensor())
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8)
        names_img = [val_dataset.samples[i][0][len(preflex) + len('validation_verification/'):] for i in range(len(val_dataset))]
        output_info = get_output(model=model, test_loader=val_dataloader, idx=idx)
        with open(preflex + 'fragment_validation_verification.txt', 'r') as f:
            for i in f.readlines():
                tmparr = i.split(" ")
                img1, img2, ref = tmparr[0], tmparr[1], tmparr[2]
                pos1, pos2 = names_img.index(img1), names_img.index(img2)
                ans = 1 - spatial.distance.cosine(output_info[pos1].detach().numpy(), output_info[pos2].detach().numpy())
                namearr.append((img1, img2))
                final.append(ans)
    with open("hw2p2_verification_sub.csv", 'w+') as f:
        f.write('trail,score\n')
        for i, j in zip(namearr, final):
            f.write(str(i[0] + '\n' + i[1]) + ',' + str(j) + '\n')
