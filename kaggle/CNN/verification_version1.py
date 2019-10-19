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
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def getitem(datasets.ImageFolder):


def parse_data(datadir):
    img_list = []
    ID_list = []
    for root, directories, filenames in os.walk(datadir):
        for filename in sorted(filenames):
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                img_list.append(filei)
                ID_list.append(filename.split('.')[0])


    # construct a dictionary, where key and value correspond to ID and target
    # uniqueID_list = list(set(ID_list))
    # class_n = len(uniqueID_list)
    # target_dict = dict(zip(uniqueID_list, range(class_n)))
    # label_list = [target_dict[ID_key] for ID_key in ID_list]
    # print('{}\t\t{}\n{}\t\t{}'.format('#Images', '#Labels', len(img_list), len(set(label_list))))
    return img_list, ID_list


def get_output(model, test_loader, idx):
    model.eval()
    output = []
    print("done")
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
        if batch_num >= 10:
            return output
    return output


if __name__ == "__main__":
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    model_option = input("Model type? (b/r): ")
    preflex = "data.nosync/"
    # if model_option == 'b':
    #     namearr = []
    #     final = []
    #     idx = 0
    #     model = B.network
    #     model.load_state_dict(torch.load('saved_models/basics/5437.pt', map_location=device))
    #     val_dataset = torchvision.datasets.ImageFolder(root=preflex + 'validation_verification', transform=torchvision.transforms.ToTensor())
    #     val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8)
    #     names_img = [val_dataset.samples[i][0][len(preflex) + len('validation_verification/'):] for i in range(len(val_dataset))]
    #     output_info = get_output(model=model, test_loader=val_dataloader, idx=idx)
    #     with open(preflex + 'fragment_validation_verification.txt', 'r') as f:
    #         for i in f.readlines():
    #             tmparr = i.split(" ")
    #             img1, img2, ref = tmparr[0], tmparr[1], tmparr[2]
    #             pos1, pos2 = names_img.index(img1), names_img.index(img2)
    #             ans = 1 - spatial.distance.cosine(output_info[pos1].detach().numpy(), output_info[pos2].detach().numpy())
    #             namearr.append((img1, img2))
    #             final.append(ans)
    # with open("hw2p2_verification_sub.csv", 'w+') as f:
    #     f.write('trail,score\n')
    #     for i, j in zip(namearr, final):
    #         f.write(str(i[0] + ' ' + i[1]) + ',' + str(j) + '\n')
    if model_option == 'b':
        namearr = []
        final = []
        idx = 0
        model = B.network
        model.load_state_dict(torch.load('saved_models/basics/5437.pt', map_location=device))

        testpath = preflex + 'test_verification'
        # img_pil = Image.open(testpath)
        img_list, id_list = parse_data(testpath)
        # print(img_list)
        model.eval()
        for i in img_list:
            img = Image.open(i)
            t = torchvision.transforms.ToTensor()(img)
            # print(t)
            _,cur_result = model(t.reshape(1,3,32,32))
            print(cur_result)
            # print(t)
            exit(1)

    #     test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)
    #     names_img = [test_dataset.samples[i][0][len(preflex) + len('validation_verification/'):] for i in range(len(test_dataset))]
    #     output_info = get_output(model=model, test_loader=test_dataloader, idx=idx)
    #     with open(preflex + 'fragment_validation_verification.txt', 'r') as f:
    #         for i in f.readlines():
    #             tmparr = i.split(" ")
    #             img1, img2, ref = tmparr[0], tmparr[1], tmparr[2]
    #             pos1, pos2 = names_img.index(img1), names_img.index(img2)
    #             ans = 1 - spatial.distance.cosine(output_info[pos1].detach().numpy(), output_info[pos2].detach().numpy())
    #             namearr.append((img1, img2))
    #             final.append(ans)
    # with open("hw2p2_verification_sub.csv", 'w+') as f:
    #     f.write('trail,score\n')
    #     for i, j in zip(namearr, final):
    #         f.write(str(i[0] + ' ' + i[1]) + ',' + str(j) + '\n')
