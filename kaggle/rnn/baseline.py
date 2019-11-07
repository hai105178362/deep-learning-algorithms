import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# import decode

# import helper.phoneme_list as PL


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
HIDDEN_SIZE = 256
# HIDDEN_SIZE = 128


class Model(torch.nn.Module):
    def __init__(self, in_vocab, out_vocab, hidden_size):
        super(Model, self).__init__()
        self.lstm = torch.nn.LSTM(in_vocab, hidden_size, bidirectional=True, num_layers=3)
        # self.lstm = torch.nn.LSTM(in_vocab, hidden_size, bidirectional=True, num_layers=3,dropout=0.2)
        # self.lstm = torch.nn.LSTM(in_vocab, hidden_size, bidirectional=True)
        self.output = torch.nn.Linear(hidden_size * 2, out_vocab)

    def forward(self, X, lengths):
        self.lstm.to(DEVICE)
        self.output.to(DEVICE)
        X = torch.nn.utils.rnn.pad_sequence(X).to(DEVICE)
        packed_X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, enforce_sorted=False).to(DEVICE)
        packed_out = self.lstm(packed_X)[0]
        out, out_lens = torch.nn.utils.rnn.pad_packed_sequence(packed_out)
        out = self.output(out).log_softmax(2).to(DEVICE)
        # print(out)
        return out, out_lens


def train_epoch_packed(model, optimizer, train_loader, n_epoch):
    # criterion = nn.CrossEntropyLoss(reduction="sum")  # sum instead of averaging, to take into account the different lengths
    criterion = nn.CTCLoss()
    criterion = criterion.to(DEVICE)
    batch_id = 0
    before = time.time()
    print("Training", len(train_loader), "number of batches")
    for inputs, targets in (train_loader):  # lists, presorted, preloaded on GPU
        batch_id += 1
        inputlen = torch.IntTensor([len(seq) for seq in inputs])
        targetlen = torch.IntTensor([len(seq) for seq in targets])
        outputs, outlens = model(inputs, inputlen)
        targets = torch.nn.utils.rnn.pad_sequence(targets).T
        loss = criterion(outputs, targets, outlens, targetlen)  # criterion of the concatenated output
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_id % 5 == 0:
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
    # for inputs, targets in val_loader:
    #     nwords += np.sum(np.array([len(l) for l in inputs]))
    #     batch_id += 1
    #     batch_id += 1
    #     inputlen = torch.IntTensor([len(seq) for seq in inputs])
    #     targetlen = torch.IntTensor([len(seq) for seq in targets])
    #     outputs, outlens = model(inputs, inputlen)
    #     targets = torch.nn.utils.rnn.pad_sequence(targets).T
    #     loss = criterion(outputs, targets, outlens, targetlen)  # criterion of the concatenated output
    #     val_loss += loss.item()
    # val_lpw = val_loss / nwords
    # print("\nValidation loss per word:", val_lpw)
    # print("Validation perplexity :", np.exp(val_lpw), "\n")
    if n_epoch > 0 and (n_epoch + 1) % 5 == 0:
        modelpath = "saved_models/{}.pt".format(str(jobtime) + "-" + str(n_epoch))
        torch.save(model.state_dict(), modelpath)
        print("Model saved at: {}".format(jobtime + modelpath))
    # return val_lpw
    return


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
        return line[:-1], line[1:]

    def __len__(self):
        return len(self.lines)


# collate fn lets you control the return value of each batch
# for packed_seqs, you want to return your data sorted by length
def collate_lines(seq_list):
    inputs, targets = zip(*seq_list)
    inputs = list(inputs)
    for i in range(len(inputs)):
        inputs[i] = torch.cat(inputs[i])
    return inputs, targets


if __name__ == "__main__":
    print("Net is running...")
    now = datetime.datetime.now()
    jobtime = str(now.hour) + ":" + str(now.minute)
    valxpath = "dataset.nosync/HW3P2_Data/wsj0_dev.npy"
    valypath = "dataset.nosync/HW3P2_Data/wsj0_dev_merged_labels.npy"
    trainxpath = "dataset.nosync/HW3P2_Data/wsj0_train.npy"
    trainypath = "dataset.nosync/HW3P2_Data/wsj0_train_merged_labels.npy"
    task = "va"
    if task == "train":
        xpath = trainxpath
        ypath = trainypath
    else:
        xpath = valxpath
        ypath = valypath
    X, Y = load_data(xpath=xpath, ypath=ypath)
    # valX, valY = load_data(xpath=valxpath, ypath=valypath)
    X = LinesDataset(X)
    # valX = LinesDataset(valX)
    traindata = []
    valdata = []
    for i, j in zip(X, Y):
        traindata.append((i, torch.IntTensor(j).to(DEVICE)))
    # for i, j in zip(valX, valY):
    #     valdata.append((i, torch.IntTensor(j).to(DEVICE)))

    # exit()
    train_loader = DataLoader(traindata, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_lines)
    # val_loader = DataLoader(valdata, shuffle=False, batch_size=BATCH_SIZE, collate_fn=collate_lines)
    model = Model(in_vocab=40, out_vocab=47, hidden_size=HIDDEN_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    for i in range(1000):
        print("==========Epoch {}==========".format(i + 1))
        train_epoch_packed(model, optimizer, train_loader, n_epoch=i)