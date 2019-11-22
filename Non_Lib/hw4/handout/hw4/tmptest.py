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
train_data = np.load('../dataset/wiki.train.npy', allow_pickle=True)
fixtures_pred = np.load('../fixtures/prediction.npz', allow_pickle=True)  # dev
fixtures_gen = np.load('../fixtures/generation.npy', allow_pickle=True)  # dev
fixtures_pred_test = np.load('../fixtures/prediction_test.npz', allow_pickle=True)  # test
fixtures_gen_test = np.load('../fixtures/generation_test.npy', allow_pickle=True)  # test
# fixtures_generation_test = np.load('../fixtures/prediction_test.npz', allow_pickle=True)  # test
# fixtures_generation_test = np.load('../fixtures/generation_test.npy', allow_pickle=True)  # test
vocab = np.load('../dataset/vocab.npy', allow_pickle=True)

print(fixtures_pred['inp'].shape)
# print(fixtures_pred['out'])
# print(fixtures_gen.shape)
# print(fixtures_gen.shape)
# print(fixtures_gen_test)
# inp = fixtures_pred_test['inp']
# for i in inp:
#     inputs = torch.LongTensor(i).unsqueeze(0)
#     print(inputs.shape)
