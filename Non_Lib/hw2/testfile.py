import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import multiprocessing as mtp
import traceback
import sys
from layers import Conv1D
from cnn import CNN_B
from mlp import MLP


data = np.loadtxt('data/data.asc').T.reshape(1, 24, -1)
cnn = CNN_B()
weights = np.load('weights/mlp_weights_part_b.npy' ,allow_pickle=True)
cnn.init_weights(weights)
# expected_result = np.load('autograde/res_b.npy')
result = cnn(data)


D = 24  # length of each feature vector
layer_sizes = [8 * D, 8, 16, 4]
mlp = MLP([8 * D, 8, 16, 4])
mlp.init_weights(weights)
mlpresult = mlp(data)
