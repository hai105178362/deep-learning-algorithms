import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# import helper.phoneme_list as PL


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Model(torch.nn.Module):
    def __init__(self, in_vocab, out_vocab, embed_size, hidden_size):
        super(Model, self).__init__()
        self.embed_size = embed_size
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.output = torch.nn.Linear(hidden_size * 2, out_vocab)

    def forward(self, X, lengths):
        X = torch.nn.utils.rnn.pad_sequence(X)
        xlens = torch.Tensor([len(X) for _ in range(5)])
        packed_X = torch.nn.utils.rnn.pack_padded_sequence(X, xlens, enforce_sorted=False)
        packed_out = self.lstm(packed_X)[0]
        out, out_lens = torch.nn.utils.rnn.pad_packed_sequence(packed_out)
        out = self.output(out).log_softmax(2)
        # print(out)
        return out,out_lens


def train_epoch_packed(model, optimizer, train_loader, val_loader, inputs_len):
    criterion = nn.CrossEntropyLoss(reduction="sum")  # sum instead of averaging, to take into account the different lengths
    criterion = criterion.to(DEVICE)
    batch_id = 0
    before = time.time()
    print("Training", len(train_loader), "number of batches")
    for inputs, targets in train_loader:  # lists, presorted, preloaded on GPU
        print(batch_id)
        batch_id += 1
        outputs,lens = model(inputs, inputs_len)
        print(targets[2].shape)
        exit()
        print(torch.cat(targets).shape)
        print(torch.cat(outputs).shape)
        exit()
        loss = criterion(torch.cat(outputs), torch.cat(targets))  # criterion of the concatenated output
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_id % 100 == 0:
            after = time.time()
            nwords = np.sum(np.array([len(l) for l in inputs]))
            lpw = loss.item() / nwords
            print("Time elapsed: ", after - before)
            print("At batch", batch_id)
            print("Training loss per word:", lpw)
            print("Training perplexity :", np.exp(lpw))
            before = after

    val_loss = 0
    batch_id = 0
    nwords = 0
    for inputs, targets in val_loader:
        nwords += np.sum(np.array([len(l) for l in inputs]))
        batch_id += 1
        outputs = model(inputs, inputs_len)
        loss = criterion(outputs, torch.cat(targets))
        val_loss += loss.item()
    val_lpw = val_loss / nwords
    print("\nValidation loss per word:", val_lpw)
    print("Validation perplexity :", np.exp(val_lpw), "\n")
    return val_lpw


def load_data(xpath, ypath=None):
    x = np.load(xpath, allow_pickle=True, encoding="bytes")
    if ypath != None:
        y = np.load(ypath, allow_pickle=True, encoding="bytes")
        return x, y
    return x


class LinesDataset(Dataset):
    def __init__(self, lines):
        self.lines = [torch.FloatTensor(l) for l in lines]

    def __getitem__(self, i):
        line = self.lines[i]
        return line[:-1].to(DEVICE), line[1:].to(DEVICE)

    def __len__(self):
        return len(self.lines)


# collate fn lets you control the return value of each batch
# for packed_seqs, you want to return your data sorted by length
def collate_lines(seq_list):
    inputs, targets = zip(*seq_list)
    lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i] for i in seq_order]
    targets = [targets[i] for i in seq_order]
    return inputs, targets


if __name__ == "__main__":
    devxpath = "dataset.nosync/HW3P2_Data/wsj0_dev.npy"
    # devxpath = "/content/drive/My Drive/datasets/hw3p2/wsj0_dev.npy"
    devypath = "dataset.nosync/HW3P2_Data/wsj0_dev_merged_labels.npy"
    # devypath = "/content/drive/My Drive/datasets/hw3p2/wsj0_dev_merged_labels.npy"
    trainxpath = "dataset.nosync/HW3P2_Data/wsj0_train.npy"
    trainypath = "dataset.nosync/HW3P2_Data/wsj0_train_merged_labels.npy"
    task = "dev"
    if task == "train":
        xpath = trainxpath
        ypath = trainypath
    else:
        xpath = devxpath
        ypath = devypath
    X, Y = load_data(xpath=xpath, ypath=ypath)
    X_lens = torch.Tensor([len(seq) for seq in X])
    print(X_lens[:5])
    Y_lens = torch.Tensor([len(seq) for seq in Y])
    print(type(X), type(Y))
    X = LinesDataset(X)
    train_loader = DataLoader(X, shuffle=False, batch_size=5, collate_fn=collate_lines)
    model = Model(in_vocab=40, out_vocab=46, embed_size=40, hidden_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    for i in range(20):
        train_epoch_packed(model, optimizer, train_loader, train_loader, X_lens)
