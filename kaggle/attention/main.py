import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as utils
import pickle as pk
from torch.utils.data import DataLoader, Dataset
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#########################
from params import config
import utils
import net


def train(model, train_loader, num_epochs, criterion, optimizer):
    for epochs in range(num_epochs):
        loss_sum = 0
        since = time.time()
        print("training...")
        for (batch_num, collate_output) in enumerate(train_loader):
            with torch.autograd.set_detect_anomaly(True):

                speech_input, text_input, speech_len, text_len = collate_output
                speech_input = speech_input.to(device)
                text_input = text_input.to(device)

                predictions = model(speech_input, speech_len, text_input)
                mask = torch.zeros(text_input.size()).to(device)

                for length in text_len:
                    mask[:, :length] = 1

                mask = mask.view(-1).to(device)

                predictions = predictions.contiguous().view(-1, predictions.size(-1))
                text_input = text_input.contiguous().view(-1)

                loss = criterion(predictions, text_input)
                masked_loss = torch.sum(loss * mask)

                masked_loss.backward()

                torch.nn.utils.clip_grad_norm(model.parameters(), 2)
                optimizer.step()

                current_loss = float(masked_loss.item()) / int(torch.sum(mask).item())

                if batch_num % 25 == 1:
                    print('train_loss', current_loss)


def main():
    model = net.Seq2Seq(input_dim=40, vocab_size=len(utils.letter_list), hidden_dim=config.hidden_dim)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    train(model=model, train_loader=utils.train_loader, num_epochs=config.num_epochs, criterion=criterion, optimizer=optimizer)


if __name__ == "__main__":
    main()
