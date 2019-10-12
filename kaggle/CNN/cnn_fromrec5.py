import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from skimage import io

#
# input_image = autograd.Variable(torch.randn(1,3,32,32)) # single 32x32 RGB image
# print(input_image.size())
# input_signal = autograd.Variable(torch.randn(1,40,100)) # 40 dimensional signal for 100 timesteps
# print(input_signal.size())
#
# # Create layers
# layer_c2d = torch.nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, stride=1, padding=2)
# layer_avg = torch.nn.AvgPool2d(kernel_size=32)
#
# # Use layers directly
# y = layer_c2d(input_image)
# print(y.size())
#
# # Add layers to model
# model = torch.nn.Sequential(layer_avg, layer_c2d)
# y = model(input_image)
# print(y.size())
#
# filters = autograd.Variable(torch.randn(20,3,5,5)) # 5x5 filter from 3 dimensions to 20
# y=F.conv2d(input_image, filters, padding=2)
# print(y.size())
#
# filters = autograd.Variable(torch.randn(256,40,5)) # 5 wide filter from 40 dimensions to 256
# y=F.conv1d(input_signal, filters, padding=2)
# print(y.size())
#
# ####################READ DATA#######################
# image = io.imread('data.nosync/11785-f19-hw2p2-classification/11-785hw2p2-f19/train_data/medium/0/0002_01.jpg')
# print(image.shape)
# # Display an image
# plt.imshow(image)
# plt.show()


# Use torch ImageFolder
# data_transform = transforms.Compose([
#         transforms.RandomResizedCrop(100),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor()
#     ])
# face_dataset = datasets.ImageFolder(root='data.nosync/11785-f19-hw2p2-classification/11-785hw2p2-f19/train_data/medium',
#                                            transform=data_transform)
# dataset_loader = torch.utils.data.DataLoader(face_dataset,
#                                              batch_size=2,
#                                              shuffle=True,
#                                              num_workers=1)
# for x,y in dataset_loader:
#     x = x.numpy()
#     print(x.shape)
#     for img, label in zip(x,y):
#         print("Label: {}".format(label))
#         plt.imshow(img.transpose((1,2,0)))
#         plt.show()


class BasicCNNModule(nn.Module):
    def __init__(self):
        super(BasicCNNModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(Relu(self.conv1(x)))
        x = self.pool(Relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = Relu(self.fc1(x))
        x = Relu(self.fc2(x))
        x = self.fc3(x)
        return x


# print(BasicCNNModule())

args,kwargs = [], {}
model = BasicCNNModule()
# save only model parameters
PATH = "saved_models/basic_cnn.pt"
torch.save(model.state_dict(), PATH)

# load a saved model parameters
model = BasicCNNModule()
model.load_state_dict(torch.load(PATH))

## Less optimised approaches ->
# saving the entire model
torch.save(model, PATH)

# load the saved model


import torchvision.models as models

vgg16 = models.vgg16(pretrained=True) # this might take a while
vgg16.eval()
print(vgg16)