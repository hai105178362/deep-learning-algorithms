import numpy as np
import torch
import params as par
from params import config
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as utils
import pickle as pk
from torch.utils.data import DataLoader, Dataset
import time
from torch.autograd import Variable
import data_utility as du
from torchnlp.nn import LockedDropout

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
        i, j, k = inp_shape[0], inp_shape[1], inp_shape[2]
        if j == 1:
            pass
        elif j % 2 != 0:
            inp = inp[:, :j - 1, :]
        inp = inp.reshape(i, j // 2, k * 2)
        inp = torch.transpose(inp, 1, 0)
        output, hidden = self.blstm(inp)
        return output, hidden


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, value_size=128, key_size=128):
        super(Encoder, self).__init__()
        # self.embed = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True).to(device)
        self.dropout = nn.Dropout(p=0.3)
        # Here you need to define the blocks of pBLSTMs
        self.pblstm1 = pBLSTM(input_dim=hidden_dim * 2, hidden_dim=hidden_dim)
        self.pblstm2 = pBLSTM(input_dim=hidden_dim * 2, hidden_dim=hidden_dim)
        self.pblstm3 = pBLSTM(input_dim=hidden_dim * 2, hidden_dim=hidden_dim)

        self.key_network = nn.Linear(hidden_dim * 2, value_size).to(device)
        self.value_network = nn.Linear(hidden_dim * 2, key_size).to(device)

        self.locked_dropouts = LockedDropout(p=0.2)

    def forward(self, x, seq_len):
        rnn_inp = utils.rnn.pack_padded_sequence(x, lengths=seq_len, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(rnn_inp)
        outputs, _ = utils.rnn.pad_packed_sequence(outputs)
        # outputs = self.dropout(outputs)
        # Use the outputs and pass it through the pBLSTM blocks
        outputs, _ = self.pblstm1(outputs)
        outputs = self.locked_dropouts(outputs)

        outputs, _ = self.pblstm2(outputs)
        # outputs = self.locked_dropouts(outputs)

        linear_input, _ = self.pblstm3(outputs)
        # linear_input = self.dropout(linear_input)

        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)
        out_seq_sizes = [size // 8 for size in seq_len]

        return keys, value, out_seq_sizes


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=1).to(device)

    def forward(self, query, key, value, speech_len):
        '''
        :param speech_len:
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
        for i, size in enumerate(speech_len):  # It should be speech len
            mask[i, :size] = 1
        # attention_score = self.softmax(energy)
        # attention_score = mask * attention_score
        attention_score = mask * energy
        attention_score = self.softmax(attention_score)
        attention_score = attention_score / torch.sum(attention_score, dim=1).unsqueeze(1).expand_as(attention_score)
        context = torch.bmm(attention_score.unsqueeze(1), value).squeeze(dim=1)
        return context, mask


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=True, embed_dim=config.embed_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0).to(device)

        self.lstm1 = nn.LSTMCell(input_size=embed_dim + value_size, hidden_size=hidden_dim).to(device)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size).to(device)
        self.dropout = nn.Dropout(p=0.1)
        # self.dropout2 = nn.Dropout(p=0.1)
        self.isAttended = isAttended
        if (isAttended):
            self.attention = Attention()
        self.character_prob = nn.Linear(key_size + value_size, vocab_size).to(device)

        self.hidden_init = torch.nn.ParameterList()
        self.char_init = torch.nn.ParameterList()
        for i in range(2):
            self.hidden_init.append(torch.nn.Parameter(torch.rand(1, hidden_dim)))
            self.char_init.append(torch.nn.Parameter(torch.rand(1, hidden_dim)))

    def forward(self, key, values, speech_len, text=None, train=par.train_mode, teacher_forcing_rate=par.tf_rate):
        '''
        :param speech_len:
        :param key :(T,N,key_size) Output of the Encoder Key projection layer
        :param values: (T,N,value_size) Output of the Encoder Value projection layer
        :param text: (N,text_len) Batch input of text with text_length
        :param train: Train or eval mode
        :return predictions: Returns the character perdiction probability
        '''
        if text is None:
            teacher_forcing_rate = 0
        teacher_force = True if np.random.random_sample() < teacher_forcing_rate else False

        batch_size = key.shape[1]

        if (train):
            max_len = text.shape[1]
            # embeddings = self.embedding(text)
        else:
            # mu, beta = 250, 5  # location and scale
            # max_len = int(np.random.gumbel(mu, beta))
            max_len = 250

        predictions = []
        hidden_states = [None, None]
        prediction = torch.zeros(batch_size, 1).to(device)
        state, pred_word = self.init_state(batch_size)
        context = values[0, :, :]
        char_embed = self.embedding(pred_word)

        for i in range(max_len):
            '''
            Here you should implement Gumble noise and teacher forcing techniques
            '''
            if (train):
                if teacher_force and i > 0:
                    pred_word = text[:, i - 1]
                    # char_embed = embeddings[:, i, :]
                    char_embed = self.embedding(pred_word)
                else:
                    pred_word = prediction.argmax(dim=-1)
                    char_embed = self.embedding(pred_word)
            else:
                pred_word = prediction.argmax(dim=-1)
                char_embed = self.embedding(pred_word)
            char_embed = self.dropout(char_embed)
            context = self.dropout(context)
            inp = torch.cat([char_embed, context], dim=1)
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            output = hidden_states[1][0]

            if self.isAttended:
                context, mask = self.attention(output, key, values, speech_len=speech_len)
                prediction = self.character_prob(torch.cat([output, context], dim=1))
            # When attention is True you should replace the values[i,:,:] with the context you get from attention
            else:
                inp = torch.cat([char_embed, values[i, :, :]], dim=1)
                prediction = self.character_prob(torch.cat([output, values[i, :, :]], dim=1))
            predictions.append(prediction.unsqueeze(1))
        return torch.cat(predictions, dim=1)

    def init_state(self, batch_size=config.batch_size):
        hidden = [h.repeat(batch_size, 1) for h in self.hidden_init]
        cell = [c.repeat(batch_size, 1) for c in self.char_init]
        output_word = Variable(hidden[0].data.new(batch_size).long().fill_(du.letter_list.index('<sos>') + 1)).to(device)
        return [hidden, cell], output_word


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, vocab_size, encode_hidden, decode_hidden, value_size=128, key_size=128):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, encode_hidden)
        self.decoder = Decoder(vocab_size + 1, decode_hidden)

    def forward(self, speech_input, speech_len, text_input=None, text_len=None, train=par.train_mode):
        key, value, seq_len = self.encoder(speech_input, speech_len)
        if train:
            predictions = self.decoder(key, value, speech_len=speech_len, text=text_input)
        else:
            predictions = self.decoder(key, value, text=None, train=False, speech_len=speech_len)
            # predictions = (predictions[:, :, 1:])
            # print(predictions[0][0])
            # exit()
        return predictions
