import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as utils
import pickle as pk
from torch.utils.data import DataLoader, Dataset
import time
import string

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from params import config

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
    start_idx = 0
    end_idx = len(letter_list) + 1
    letter_to_index_list = []
    # print(letter_to_index_list)
    for n, i in enumerate(transcript):
        cur_sentence = np.array([0])
        for j in i:
            cur_word = str(j)[2:-1]
            word_idx = np.array([letter_list.index(' ')] + [letter_list.index(char) + 1 for char in cur_word])
            cur_sentence = np.append([cur_sentence], [word_idx])
        letter_to_index_list.append(cur_sentence)
    return np.array(letter_to_index_list)


class Speech2Text_Dataset(Dataset):
    def __init__(self, speech, text=None, train=True):
        self.speech = speech
        self.train = train
        if (text is not None):
            self.text = text

    def __len__(self):
        return len(self.speech)

    def __getitem__(self, index):
        if (self.train):
            return (self.speech[index]), (self.text[index])
        else:
            return self.speech[index]


def collate_train(batch_data):
    inputs, targets = zip(*batch_data)
    inputs_len = torch.IntTensor([len(_) for _ in inputs])
    targets_len = torch.IntTensor([len(_) for _ in targets])
    inputs = torch.nn.utils.rnn.pad_sequence(inputs)
    targets = torch.nn.utils.rnn.pad_sequence(targets)
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
    return


def generate_train_data(x, y):
    tx = [torch.FloatTensor(_) for _ in x]
    ty = [torch.LongTensor(_) for _ in y]
    # tx, ty = [(torch.DoubleTensor(i), torch.IntTensor(j)) for i, j in zip(x, y)]
    return tx, ty
    # print(train_data[0])


utterance, transcript = get_data()
letter_to_index_list = transform_letter_to_index(transcript)
tx, ty = generate_train_data(utterance, letter_to_index_list)
train_dataset = Speech2Text_Dataset(speech=tx, text=ty, train=True)
# print(train_dataset.__getitem__(0))
# train_loader = DataLoader(train_dataset, shuffle=False, batch_size=config.train_batch_size, collate_fn=collate_train)
train_loader = DataLoader(train_dataset, shuffle=False, batch_size=config.train_batch_size, collate_fn=collate_train)
if __name__ == "__main__":
    for batch_num, (inputs, targets, inputs_len, targets_len) in enumerate(train_loader):
        print(inputs)
        print(inputs.shape)
        print(targets.shape)
        print(inputs_len, targets_len)
        exit()
    print(letter_to_index_list[0])
