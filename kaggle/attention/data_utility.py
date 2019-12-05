import numpy as np
from params import config
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as utils
import pickle as pk
from torch.utils.data import DataLoader, Dataset
import time
import string
import params as par

device = 'cuda' if torch.cuda.is_available() else 'cpu'

letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ', '<sos>', '<eos>']


def get_data(mode=config.mode):
    print("Mode:{}".format(config.mode))
    if mode == "train":
        speech = np.load(config.path_train_new, allow_pickle=True, encoding='bytes')
        transcript = np.load(config.path_train_transcripts, allow_pickle=True, encoding='bytes')
    if mode == "dev":
        speech = np.load(config.path_dev_new, allow_pickle=True, encoding='bytes')
        transcript = np.load(config.path_dev_transcripts, allow_pickle=True, encoding='bytes')
    else:
        speech = np.load(config.path_test_new, allow_pickle=True, encoding='bytes')
        transcript = None
    print("Data Loading Sucessful.....")
    return speech, transcript


def transform_letter_to_index(transcript):
    '''
    :param transcript :(N, ) Transcripts are the text input
    :param letter_list: Letter list defined above
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''
    letter_to_index_list = []
    for n, i in enumerate(transcript):
        cur_sentence = []
        for n2, j in enumerate(i):
            cur_word = str(j)[2:-1]
            if n2 == 0:
                word_idx = np.array([letter_list.index('<sos>')] + [letter_list.index(char) for char in cur_word])
            elif n2 == len(i) - 1:
                word_idx = np.array([letter_list.index(' ')] + [letter_list.index(char) for char in cur_word] + [letter_list.index('<eos>')])
            else:
                word_idx = np.array([letter_list.index(' ')] + [letter_list.index(char) for char in cur_word])
            cur_sentence = np.append(cur_sentence, word_idx)
        letter_to_index_list.append(cur_sentence)
    return np.array(letter_to_index_list)


def build_vocab(transcripts):
    vocab = set()
    for i in (transcripts):
        for j in i:
            cur_word = str(j)[2:-1]
            vocab.add(cur_word)
    return ['SOS/EOS'] + [' '] + list(vocab)


def transform_word_to_index(transcripts, vocab):
    word_to_index_list = []
    for n, i in enumerate(transcripts):
        cur_sentence = [0]
        for n2, j in enumerate(i):
            cur_word = str(j)[2:-1]
            if n2 == 0:
                word_idx = [vocab.index(cur_word)]
            elif n2 == len(i) - 1:
                word_idx = np.array([vocab.index(' ')] + [vocab.index(cur_word)] + [0])
            else:
                word_idx = np.array([vocab.index(' ')] + [vocab.index(cur_word)])
            cur_sentence = np.append(cur_sentence, word_idx)
        word_to_index_list.append(cur_sentence)
    return np.array(word_to_index_list)


class Speech2Text_Dataset(Dataset):
    def __init__(self, speech, text=None, train=par.train_mode):
        self.speech = speech
        self.train = train
        if (text is not None):
            self.text = text

    def __len__(self):
        return len(self.speech)

    def __getitem__(self, index):
        if self.train:
            return (self.speech[index]), (self.text[index])
        else:
            return self.speech[index]


def collate_train(batch_data):
    inputs, targets = zip(*batch_data)

    inputs_len = torch.IntTensor([len(_) for _ in inputs])
    targets_len = torch.IntTensor([len(_) for _ in targets])
    inputs = torch.nn.utils.rnn.pad_sequence(inputs)
    targets = torch.nn.utils.rnn.pad_sequence(targets)
    targets = torch.transpose(targets, 0, 1)
    '''
    Complete this function.
    I usually return padded speech and text data, and length of
    utterance and transcript from this function
    '''
    return inputs, targets, inputs_len, targets_len


def collate_test(batch_data):
    '''
    Complete this function.
    I usually return padded speech and length of
    utterance from this function
    '''
    inputs_len = torch.IntTensor([len(_) for _ in batch_data])
    inputs = torch.nn.utils.rnn.pad_sequence(batch_data)

    return inputs, inputs_len


def generate_data(x, y=None):
    if par.train_mode:
        tx = [torch.FloatTensor(_) for _ in x]
        ty = [torch.LongTensor(_) for _ in y]
        return tx, ty
    else:
        return [torch.FloatTensor(_) for _ in x]


vocab = letter_list
if par.train_mode:
    utterance, transcript = get_data()
    # vocab = build_vocab(transcript)
    letter_to_index_list = transform_letter_to_index(transcript)
    # word_to_index_list = transform_word_to_index(transcripts=transcript, vocab=vocab)
    tx, ty = generate_data(x=utterance, y=letter_to_index_list)
    train_dataset = Speech2Text_Dataset(speech=tx, text=ty)
    # data_loader = DataLoader(train_dataset, shuffle=par.train_mode, batch_size=config.train_batch_size, collate_fn=collate_train)
    data_loader = DataLoader(train_dataset, shuffle=False, batch_size=config.train_batch_size, collate_fn=collate_train)
else:
    utterance, _ = get_data()
    x = generate_data(x=utterance)
    test_dataset = Speech2Text_Dataset(speech=x)
    data_loader = DataLoader(test_dataset, shuffle=par.train_mode, batch_size=config.test_batch_size, collate_fn=collate_test)

if __name__ == "__main__":
    for batch_num, (inputs, targets, inputs_len, targets_len) in enumerate(data_loader):
        # print(inputs)
        # print(inputs.shape)
        print(targets)
        print([letter_list[i] for i in targets[1]])
        print(inputs_len, targets_len)
        exit()
    print(letter_to_index_list[0])
