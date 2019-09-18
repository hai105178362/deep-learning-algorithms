import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


class generate_data(object):
    def __init__(self):
        super(generate_data, self).__init__()
        self.dev_labels = (
                              np.load("/Users/robert/Downloads/11-785hw1p2-f19/dev_labels.npy", allow_pickle=True))[:5]
        self.dev = np.array(np.load("/Users/robert/Downloads/11-785hw1p2-f19/dev.npy", allow_pickle=True))[:5]
        self.test = (np.load("/Users/robert/Downloads/11-785hw1p2-f19/test.npy", allow_pickle=True))
        self.train_labels = np.array(
            np.load("/Users/robert/Downloads/11-785hw1p2-f19/train_labels.npy", allow_pickle=True))[:5]
        self.train = np.array(np.load("/Users/robert/Downloads/11-785hw1p2-f19/train.npy", allow_pickle=True))[:5]


def training_routine(net, dataset, n_iters, gpu):
    # organize the data
    train_data, train_labels, val_data, val_labels = dataset

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    # use the flag
    # train_data, train_labels = train_data, train_labels.long()
    # val_data, val_labels = val_data, val_labels.long()
    if gpu:
        train_data, train_labels = train_data.cuda(), train_labels.cuda()
        val_data, val_labels = val_data.cuda(), val_labels.cuda()
        net = net.cuda()  # the network parameters also need to be on the gpu !
        print("Using GPU")
    else:
        train_data, train_labels = train_data.cpu(), train_labels.cpu()
        val_data, val_labels = val_data.cpu(), val_labels.cpu()
        net = net.cpu()  # the network parameters also need to be on the gpu !
        print("Using CPU")
    for i in range(n_iters):
        # forward pass
        train_output = net(train_data.float())
        train_loss = criterion(train_output, train_labels)
        # backward pass and optimization
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Once every 100 iterations, print statistics
        # if i % 100 == 0:
        print("At iteration", i)
        # compute the accuracy of the prediction
        train_prediction = train_output.cpu().detach().argmax(dim=1)
        train_accuracy = (train_prediction.cpu().numpy() == train_labels.cpu().numpy()).mean()
        # Now for the validation set
        val_output = net(val_data)
        val_loss = criterion(val_output, val_labels)
        # compute the accuracy of the prediction
        val_prediction = val_output.cpu().detach().argmax(dim=1)
        val_accuracy = (val_prediction.cpu().numpy() == val_labels.cpu().numpy()).mean()
        print("Training loss :", train_loss.cpu().detach().numpy())
        print("Training accuracy :", train_accuracy)
        print("Validation loss :", val_loss.cpu().detach().numpy())
        print("Validation accuracy :", val_accuracy)

    net = net.cpu()


f = generate_data()
print("{}{}{}{}".format(f.train[0].shape, f.train_labels[0].shape, f.dev[0].shape, f.dev_labels[0].shape))
dataset = torch.from_numpy(f.train[0]), torch.from_numpy(f.train_labels[0]), torch.from_numpy(
    f.dev[0]), torch.from_numpy(f.dev_labels[0])


def generate_single_hidden_MLP(n_hidden_neurons):
    return nn.Sequential(nn.Linear(40, n_hidden_neurons), nn.ReLU(), nn.Linear(n_hidden_neurons, 1))


model1 = generate_single_hidden_MLP(6)
training_routine(model1, dataset, 5, gpu=False)
n_in, n_h, n_out, batch_size = 10, 5, 40, 477
x = dataset[0]
y = dataset[1]
model = nn.Sequential(nn.Linear(n_in, n_h),
                      nn.ReLU(),
                      nn.Linear(n_h, n_out),
                      nn.Sigmoid())
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(50):
    # Forward Propagation
    y_pred = model(x)
    # Compute and print loss
    loss = criterion(y_pred, y)
    print('epoch: ', epoch, ' loss: ', loss.item())
    # Zero the gradients
    optimizer.zero_grad()

    # perform a backward pass (backpropagation)
    loss.backward()

    # Update the parameters
    optimizer.step()
print(y_pred, y)
