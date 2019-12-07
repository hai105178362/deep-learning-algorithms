import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as utils
import pickle as pk
from torch.utils.data import DataLoader, Dataset
import time
import datetime
import Levenshtein

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#########################
import params as par
from params import config
import data_utility as du
import net


def train(model, train_loader, val_loader, num_epochs, criterion, optimizer):
    best_loss = 0.1
    # model.load_state_dict(state_dict=torch.load('snapshots/{}.pt'.format(config.model), map_location=net.device))
    for epochs in range(num_epochs):
        par.tf_rate *= 0.8
        loss_sum = 0
        since = time.time()
        print("\n\n")
        print("----------------Epoch {}------------------------".format(epochs))
        print("----------------Train----------------------------")
        for (batch_num, collate_output) in enumerate(train_loader):
            with torch.autograd.set_detect_anomaly(True):
                speech_input, text_input, speech_len, text_len = collate_output
                speech_input = speech_input.to(device)
                text_input = text_input.to(device)
                predictions = model(speech_input, speech_len, text_input, text_len=text_len)
                mask = torch.zeros(text_input.size()).to(device)
                for length in text_len:
                    mask[:, :length] = 1
                mask = mask.view(-1).to(device)
                predictions = predictions.contiguous().view(-1, predictions.size(-1))

                text_input = text_input.contiguous().view(-1)

                loss = criterion(predictions, text_input)
                masked_loss = torch.sum(loss * mask)

                masked_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()

                current_loss = float(masked_loss.item()) / int(torch.sum(mask).item())
                if batch_num % 20 == 0:
                    pred2words = torch.argmax(predictions, dim=1)
                    print(text_input[:].detach().cpu().numpy())
                    print(pred2words[:].data.detach().cpu().numpy())
                    ref = ''.join([du.letter_list[i] for i in text_input])
                    gen = ''.join([du.letter_list[i] for i in pred2words])
                    print("Batch {} Loss: {}    Levenshtein:{}".format(batch_num, current_loss, Levenshtein.distance(ref, gen)))
                    # if current_loss < best_loss:
                    #     now = datetime.datetime.now()
                    #     jobtime = str(now.hour) + str(now.minute)
                    #     modelpath = "snapshots/{}.pt".format(str(jobtime) + "-" + str(epochs))
                    #     torch.save(model.state_dict(), modelpath)
                    #     print("model saved at: ", "snapshots/{}.pt".format(str(jobtime) + "-" + str(epochs)))
                    #     best_loss = current_loss
        print("----------------Validation------------------------")
        val_loss = 0
        for (batch_num, collate_output) in enumerate(val_loader):
            speech_input, text_input, speech_len, text_len = collate_output
            speech_input = speech_input.to(device)
            text_input = text_input.to(device)
            predictions = model(speech_input, speech_len, train=False)
            mask = torch.zeros(text_input.size()).to(device)
            for length in text_len:
                mask[:, :length] = 1
            mask = mask.view(-1).to(device)
            # print(predictions.shape)
            predictions = predictions[:, :text_input.shape[1], :]
            predictions = predictions.contiguous().view(-1, predictions.size(-1))
            text_input = text_input.contiguous().view(-1)
            loss = criterion(predictions, text_input)
            masked_loss = torch.sum(loss * mask)
            val_loss += float(masked_loss.item()) / int(torch.sum(mask).item())
            if batch_num % 20 == 0:
                pred2words = torch.argmax(predictions, dim=1)
                print(text_input[:].detach().cpu().numpy())
                print(pred2words[:].data.detach().cpu().numpy())
                ref = ''.join([du.letter_list[i] for i in text_input[:min(250, len(text_input) - 1)]])
                gen = ''.join([du.letter_list[i] for i in pred2words[:min(250, len(pred2words) - 1)]])
                print("Batch: {} Levenshtein: {} ".format(batch_num, Levenshtein.distance(ref, gen)))
                # print("current_loss: {}".format(current_loss))
        print("Validation Loss: {}".format(val_loss / len(val_loader)))


def test(model, test_loader):
    with torch.no_grad():
        model.eval()
        for (batch_num, collate_output) in enumerate(test_loader):
            speech_input, speech_len = collate_output
            speech_input = speech_input.to(device)
            predictions = model(speech_input, speech_len, train=False)
            mask = torch.zeros(text_input.size()).to(device)
            # for length in text_len:
            mask[:, :250] = 1
            mask = mask.view(-1).to(device)
            # print(predictions.shape)
            predictions = predictions[:, :text_input.shape[1], :]
            predictions = predictions.contiguous().view(-1, predictions.size(-1))
            pred2words = torch.argmax(predictions, dim=1)
            print(pred2words[:].data.detach().cpu().numpy())
            text_input = [x for x in text_input if x != 0]
            pred2words = [x for x in pred2words if x != 0]
            gen = ''.join([du.letter_list[i - 1] for i in pred2words[:min(251, len(pred2words) - 1)]])


def main():
    model = net.Seq2Seq(input_dim=40, vocab_size=len(du.vocab), decode_hidden=config.decode_hidden, encode_hidden=config.encode_hidden)
    criterion = nn.CrossEntropyLoss(reduce=None).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if par.train_mode:
        train(model=model, train_loader=du.train_loader, val_loader=du.val_loader, num_epochs=config.num_epochs, criterion=criterion, optimizer=optimizer)
    else:
        test(model=model, data_loader=du.test_loader)


if __name__ == "__main__":
    main()
