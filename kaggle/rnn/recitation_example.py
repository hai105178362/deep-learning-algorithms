# Words with only E, I, N, S, T.
# Pronunciation is from http://www.speech.cs.cmu.edu/cgi-bin/pronounce
data = [
    ('SEE', 'S IY'),
    ('SET', 'S EH T'),
    ('SIT', 'S IH T'),
    ('SITE', 'S AY T'),
    ('SIN', 'S IH N'),
    ('TEEN', 'T IY N'),
    ('TIN', 'T IH N'),
    ('TIE', 'T AY'),
    ('TEST', 'T EH S T'),
    ('NET', 'N EH T'),
    ('NEET', 'N IY T'),
    ('NINE', 'N AY N')
]
letters = 'EINST'
# Starts with ' ' for blank, followed by actual phonemes
phonemes = [' ', 'S', 'T', 'N', 'IY', 'IH', 'EH', 'AY']

import torch
from torch import nn
from torch.nn.utils.rnn import *

X = [torch.LongTensor([letters.find(c) for c in word]) for word, _ in data]
Y = [torch.LongTensor([phonemes.index(p) for p in pron.split()]) for _, pron in data]
X_lens = torch.LongTensor([len(seq) for seq in X])
Y_lens = torch.LongTensor([len(seq) for seq in Y])
X = pad_sequence(X)
# `batch_first=True` is required for use in `nn.CTCLoss`.
Y = pad_sequence(Y, batch_first=True)

print('X', X.size(), X_lens)
print('Y', Y.size(), Y_lens)


class Model(nn.Module):
    def __init__(self, in_vocab, out_vocab, embed_size, hidden_size):
        super(Model, self).__init__()
        self.embed = nn.Embedding(in_vocab, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.output = nn.Linear(hidden_size * 2, out_vocab)

    def forward(self, X, lengths):
        X = self.embed(X)
        packed_X = pack_padded_sequence(X, lengths, enforce_sorted=False)
        packed_out = self.lstm(packed_X)[0]
        out, out_lens = pad_packed_sequence(packed_out)
        # Log softmax after output layer is required for use in `nn.CTCLoss`.
        out = self.output(out).log_softmax(2)
        return out, out_lens

torch.manual_seed(11785)
model = Model(len(letters), len(phonemes), 4, 4)
criterion = nn.CTCLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(50):
    model.zero_grad()
    out, out_lens = model(X, X_lens)
    loss = criterion(out, Y, out_lens, Y_lens)
    print('Epoch', epoch + 1, 'Loss', loss.item())
    loss.backward()
    optimizer.step()


import torch
from ctcdecode import CTCBeamDecoder

decoder = CTCBeamDecoder([' ', 'A'], beam_width=4)
probs = torch.Tensor([[0.2, 0.8], [0.8, 0.2]]).unsqueeze(0)
print(probs.size())
out, _, _, out_lens = decoder.decode(probs, torch.LongTensor([2]))
print(out[0, 0, :out_lens[0, 0]])
print(out)
# import numpy as np
# y_s = np.load('table_of_ys_brand_new.npy')
# #y_s = np.array([[1/6,4/6,2/6,1/6],[2/6,1/6,1/6,4/6],[3/6,1/6,3/6,1/6]])
# #y_s = y_s.reshape(y_s.shape[0],y_s.shape[1],1)
# y_sT = np.transpose(y_s, (2,1,0))
# tensor_y = torch.Tensor(y_sT)
# decoder = CTCBeamDecoder([' ','a','b','c'], beam_width=2)
# out, _, _, out_lens = decoder.decode(tensor_y, torch.LongTensor([10]))
# print(out[0, 0, :out_lens[0, 0]])