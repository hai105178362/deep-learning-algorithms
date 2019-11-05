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


class Model(torch.nn.Module):
    def __init__(self, in_vocab, out_vocab, hidden_size):
        super(Model, self).__init__()
        self.lstm = torch.nn.LSTM(in_vocab, hidden_size, bidirectional=True, num_layers=3,dropout=0.5)
        # self.lstm = torch.nn.LSTM(in_vocab, hidden_size, bidirectional=True)
        self.output = torch.nn.Linear(hidden_size * 2, out_vocab)

    def forward(self, X, lengths):
        self.lstm.to(DEVICE)
        self.output.to(DEVICE)
        X = torch.nn.utils.rnn.pad_sequence(X)
        xlens = torch.Tensor([len(X) for _ in range(len(lengths))]).to(DEVICE)
        packed_X = torch.nn.utils.rnn.pack_padded_sequence(X, xlens, enforce_sorted=False).to(DEVICE)
        packed_out = self.lstm(packed_X)[0]
        out, out_lens = torch.nn.utils.rnn.pad_packed_sequence(packed_out)
        out = self.output(out).log_softmax(2)
        return out, out_lens


def train_epoch_packed(model, optimizer, train_loader, val_loader, inputs_len, val_inputs_len, n_epoch):
    # criterion = nn.CrossEntropyLoss(reduction="sum")  # sum instead of averaging, to take into account the different lengths
    criterion = nn.CTCLoss()
    criterion = criterion.to(DEVICE)
    batch_id = 0
    before = time.time()
    print("Training", len(train_loader), "number of batches")
    for inputs, targets in train_loader:  # lists, presorted, preloaded on GPU
        batch_id += 1
        new_inputlen = inputs_len[(batch_id - 1) * BATCH_SIZE:batch_id * BATCH_SIZE]
        outputs, outlens = model(inputs, new_inputlen)

        cur_Y = Y[(batch_id - 1) * BATCH_SIZE:batch_id * BATCH_SIZE]
        cur_Y_len = Y_lens[(batch_id - 1) * BATCH_SIZE:batch_id * BATCH_SIZE]
        cur_Y = torch.nn.utils.rnn.pad_sequence(cur_Y).T

        loss = criterion(outputs, cur_Y, outlens, cur_Y_len)  # criterion of the concatenated output
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_id % 200 == 0:
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
        new_inputlen = val_inputs_len[(batch_id - 1) * BATCH_SIZE:batch_id * BATCH_SIZE]
        outputs, outlens = model(inputs, new_inputlen)
        cur_Y = valY[(batch_id - 1) * BATCH_SIZE:batch_id * BATCH_SIZE]
        cur_Y_len = valY_lens[(batch_id - 1) * BATCH_SIZE:batch_id * BATCH_SIZE]
        cur_Y = torch.nn.utils.rnn.pad_sequence(cur_Y).T
        loss = criterion(outputs, cur_Y, outlens, cur_Y_len)  # criterion of the concatenated output
        val_loss += loss.item()
    val_lpw = val_loss / nwords
    print("\nValidation loss per word:", val_lpw)
    print("Validation perplexity :", np.exp(val_lpw), "\n")
    if n_epoch > 0 and (n_epoch + 1) % 5 == 0:
        modelpath = "saved_models/{}.pt".format(str(jobtime) + "-" + str(n_epoch))
        torch.save(model.state_dict(), modelpath)
        print("Model saved at: {}".format(jobtime + modelpath))
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
    print("Net is running...")
    now = datetime.datetime.now()
    jobtime = str(now.hour) + ":" + str(now.minute)
    valxpath = "dataset.nosync/HW3P2_Data/wsj0_dev.npy"
    # devxpath = "/content/drive/My Drive/datasets/hw3p2/wsj0_dev.npy"
    valypath = "dataset.nosync/HW3P2_Data/wsj0_dev_merged_labels.npy"
    # devypath = "/content/drive/My Drive/datasets/hw3p2/wsj0_dev_merged_labels.npy"
    trainxpath = "dataset.nosync/HW3P2_Data/wsj0_train.npy"
    trainypath = "dataset.nosync/HW3P2_Data/wsj0_train_merged_labels.npy"
    task = "train"
    if task == "train":
        xpath = trainxpath
        ypath = trainypath
    else:
        xpath = valxpath
        ypath = valypath
    X, Y = load_data(xpath=xpath, ypath=ypath)
    valX, valY = load_data(xpath=valxpath, ypath=valypath)
    for i in range(len(Y)):
        Y[i] = torch.IntTensor(Y[i]).to(DEVICE)
    for i in range(len(valY)):
        valY[i] = torch.IntTensor(valY[i]).to(DEVICE)
    X_lens = torch.Tensor([len(seq) for seq in X]).to(DEVICE)
    print(X_lens)
    Y_lens = torch.IntTensor([len(seq) for seq in Y]).to(DEVICE)
    X = LinesDataset(X)
    valX_lens = torch.Tensor([len(seq) for seq in valX]).to(DEVICE)
    valY_lens = torch.IntTensor([len(seq) for seq in valY]).to(DEVICE)
    valX = LinesDataset(valX)

    train_loader = DataLoader(X, shuffle=False, batch_size=BATCH_SIZE, collate_fn=collate_lines)
    val_loader = DataLoader(valX, shuffle=False, batch_size=BATCH_SIZE, collate_fn=collate_lines)
    model = Model(in_vocab=40, out_vocab=47, hidden_size=196)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    for i in range(1000):
        print("==========Epoch {}==========".format(i + 1))
        train_epoch_packed(model, optimizer, train_loader, val_loader, inputs_len=X_lens, val_inputs_len=valX_lens, n_epoch=i)
