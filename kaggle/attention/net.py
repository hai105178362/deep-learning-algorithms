import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as utils
import pickle as pk
from torch.utils.data import DataLoader, Dataset
import time
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class pBLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, factor=2):
        super(pBLSTM, self).__init__()
        self.factor = factor
        self.blstm = nn.LSTM(input_size=input_dim * factor, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=False).to(device)

    def forward(self, x):
        '''
        :param x :(N,T, H1) input to the pBLSTM h
        :return output: (N,T,H) encoded sequence from pyramidal Bi-LSTM
        '''
        # inp, lens = utils.rnn.pad_packed_sequence(x)
        inp = torch.transpose(x, 0, 1)
        inp_shape = (inp.shape)
        # print(inp_shape)
        # exit()
        i, j, k = inp_shape[0], inp_shape[1], inp_shape[2]
        if j % 2 != 0:
            inp = inp[:, :j - 1, :]
        inp = inp.reshape(i, j // 2, k * 2)
        inp = torch.transpose(inp, 1, 0)
        output, hidden = self.blstm(inp)
        return output, hidden


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, value_size=128, key_size=128):
        super(Encoder, self).__init__()
        # self.embed = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=False).to(device)
        # Here you need to define the blocks of pBLSTMs
        self.pblstm = pBLSTM(input_dim=hidden_dim * 2, hidden_dim=hidden_dim).to(device)

        self.key_network = nn.Linear(hidden_dim * 2, value_size)
        self.value_network = nn.Linear(hidden_dim * 2, key_size)

    def forward(self, x, lens):
        # rnn_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=False)
        # print(x.shape)
        # print(rnn_inp.data.shape)
        # outputs, _ = self.lstm(rnn_inp)
        outputs, _ = self.lstm(x)
        # Use the outputs and pass it through the pBLSTM blocks
        linear_input, _ = self.pblstm(outputs)
        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)

        return keys, value


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=1).to(device)

    def forward(self, query, key, value, text_lens):
        '''
        :param text_lens:
        :param query :(N,context_size) Query is the output of LSTMCell from Decoder
        :param key: (T,N,key_size) Key Projection from Encoder per time step
        :param value: (T,N,value_size) Value Projection from Encoder per time step
        :return output: Attended Context
        :return attention_mask: Attention mask that can be plotted
        '''
        query = query.unsqueeze(2)
        key = key.transpose(1, 0)
        value = value.transpose(1, 0)
        energy = torch.bmm(key, query).squeeze(2)
        mask = Variable(energy.data.new(energy.size(0), energy.size(1)).zero_(), requires_grad=False)
        for i, size in enumerate(text_lens):
            mask[i, :size] = 1
        attention_score = self.softmax(energy)
        attention_score = mask * attention_score
        attention_score = attention_score / torch.sum(attention_score, dim=1).unsqueeze(1).expand_as(attention_score)
        context = torch.bmm(attention_score.unsqueeze(1), value).squeeze(dim=1)
        return context, mask


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=True):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim).to(device)

        self.lstm1 = nn.LSTMCell(input_size=hidden_dim + value_size, hidden_size=hidden_dim).to(device)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size).to(device)
        self.isAttended = isAttended
        if (isAttended):
            self.attention = Attention()
        self.character_prob = nn.Linear(key_size + value_size, vocab_size).to(device)

    def forward(self, key, values, text=None, text_lens=None, train=True):
        '''
        :param text_lens:
        :param key :(T,N,key_size) Output of the Encoder Key projection layer
        :param values: (T,N,value_size) Output of the Encoder Value projection layer
        :param text: (N,text_len) Batch input of text with text_length
        :param train: Train or eval mode
        :return predictions: Returns the character perdiction probability
        '''
        batch_size = key.shape[1]
        if (train):
            max_len = text.shape[1]
            embeddings = self.embedding(text)
        else:
            max_len = 250
        predictions = []
        hidden_states = [None, None]
        prediction = torch.zeros(batch_size, 1).to(device)
        # state, output_word = self.get_initial_state(batch_size)
        for i in range(max_len):
            '''
            Here you should implement Gumble noise and teacher forcing techniques
            '''
            if (train):
                char_embed = embeddings[:, i, :]
            else:
                char_embed = self.embedding(prediction.argmax(dim=-1))
            if self.isAttended:
                context, mask = self.attention(char_embed, key, values, text_lens)
                inp = torch.cat([char_embed, context], dim=1)
            # When attention is True you should replace the values[i,:,:] with the context you get from attention
            else:
                inp = torch.cat([char_embed, values[i, :, :]], dim=1)
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            output = hidden_states[1][0]
            prediction = self.character_prob(torch.cat([output, values[i, :, :]], dim=1))
            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim=1)

    def get_initial_state(self, batch_size=32):
        hidden = [h.repeat(batch_size, 1) for h in self.rnn_inith]
        cell = [c.repeat(batch_size, 1) for c in self.rnn_initc]
        # <sos> (same vocab as <eos>)
        output_word = Variable(hidden[0].data.new(batch_size).long().fill_(chr2idx['<eos>']))
        return [hidden, cell], output_word


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=False):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, hidden_dim)

    def forward(self, speech_input, speech_len, text_input=None, text_len=None, train=True):
        key, value = self.encoder(speech_input, speech_len)
        if (train):
            predictions = self.decoder(key, value, text_input, text_lens=text_len)
        else:
            predictions = self.decoder(key, value, text=None, train=False)
        return predictions
