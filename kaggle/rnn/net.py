from torch.nn import CTCLoss
import torch
import torch.nn as nn
from torch.nn.utils.rnn import *
import numpy as np
import sys
import helper.phoneme_list as PL
import helper
from torch.utils import data
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, in_vocab, out_vocab, embed_size, hidden_size):
        super(Model, self).__init__()
        # self.embed = nn.Embedding(in_vocab, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.output = nn.Linear(hidden_size * 2, out_vocab)

    def forward(self, X, lengths):
        # X = self.embed(X)
        X = X.type(torch.FloatTensor)
        packed_X = pack_padded_sequence(X, lengths, enforce_sorted=False)
        print(packed_X[0].shape)
        packed_out = self.lstm(packed_X)[0]
        out, out_lens = pad_packed_sequence(packed_out)
        # Log softmax after output layer is required for use in `nn.CTCLoss`.
        out = self.output(out).log_softmax(2)
        return out, out_lens