import numpy as np
from matplotlib import pyplot as plt
import time
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tests import test_prediction, test_generation
from helper import loader
import csv

dataset = np.load('../dataset/wiki.train.npy', allow_pickle=True)
fixtures_pred = np.load('../fixtures/prediction.npz', allow_pickle=True)  # dev
fixtures_gen = np.load('../fixtures/generation.npy', allow_pickle=True)  # dev
fixtures_pred_test = np.load('../fixtures/prediction_test.npz', allow_pickle=True)  # test
fixtures_gen_test = np.load('../fixtures/generation_test.npy', allow_pickle=True)  # test
vocab = np.load('../dataset/vocab.npy', allow_pickle=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class argument_parser():
    def __init__(self):
        self.shuffle = True
        self.nepoch = 10
        self.batch_size = 10


args = argument_parser()


class LanguageModelDataLoader(DataLoader):
    """
        TODO: Define data loader logic here
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        print("get it")
        # raise NotImplemented

    def __iter__(self):
        num_iters = self.dataset.__len__() // self.batch_size
        for i in range(num_iters + 1):
            sentence = []
            for j in range(self.batch_size):
                if i * self.batch_size + j < len(self.dataset):
                    sentence = np.concatenate((sentence, self.dataset[i * self.batch_size + j]), axis=None)
            yield sentence
    #     # concatenate your articles and build into batches
    #
    #     raise NotImplemented


vocab_human = []
with open('../dataset/vocab.csv') as f:
    fo = csv.reader(f, delimiter=',')
    vocab_human = np.array([i[1] for i in fo][1:])
# print(vocab_human)
# train_dataloader = DataLoader(dataset=dataset,batch_size=10,shuffle=False,num_workers=5)
print(len(dataset))
print(dataset[0])
# exit()
train_newloader = LanguageModelDataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
print(train_newloader.__len__())
for i, j in enumerate(train_newloader):
    print(i, len(j))
