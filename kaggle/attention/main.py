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
# import Levenshtein
from write_csv import run_write

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#########################
import params as par
from params import config
import data_utility as du
import net


def train(model, train_loader, val_loader, num_epochs, criterion, optimizer):
    best_loss = 7
    # model.load_state_dict(state_dict=torch.load('snapshots/{}.pt'.format(config.model), map_location=net.device))
    for epochs in range(num_epochs):
        start_time = time.time()
        par.tf_rate *= 0.9
        loss_sum = 0
        since = time.time()
        print("\n\n")
        print("Epoch {}".format(epochs))
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
                optimizer.zero_grad()

                current_loss = float(masked_loss.item()) / int(torch.sum(mask).item())
                loss_sum += current_loss

                if batch_num % 30 == 0:
                    pred2words = torch.argmax(predictions, dim=1)
                    new_text = [i for i in text_input if i != 0]
                    new_gen = [i for i in pred2words if i != 0]
                    ref = ''.join([du.letter_list[i - 1] for i in new_text])
                    gen = ''.join([du.letter_list[i - 1] for i in new_gen])
                    print("Batch {} Loss: {:3f}".format(batch_num, current_loss), " | ", ref[:40], " | ", gen[:40])

        end_time = time.time()
        print("Average Training Loss: {}".format(loss_sum / len(train_loader)))
        print("Training time: {}".format(end_time - start_time))
        start_time = end_time
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
            if predictions.shape[0] > text_input.shape[0]:
                predictions = predictions[:text_input.shape[0], :]
            elif predictions.shape[0] < text_input.shape[0]:
                text_input = text_input[:predictions.shape[0]]

            loss = criterion(predictions, text_input)
            if len(loss) != len(mask):
                mask = mask[:len(loss)]
            masked_loss = torch.sum(loss * mask)
            val_loss += float(masked_loss.item()) / int(torch.sum(mask).item())
            if batch_num % 5 == 0:
                pred2words = torch.argmax(predictions, dim=1)

                text_input_view = text_input[:].detach().cpu().numpy()
                pred2words_view = pred2words[:].data.detach().cpu().numpy()

                ref = ''.join([du.letter_list[i - 1] for i in text_input_view])
                gen = ''.join([du.letter_list[i - 1] for i in pred2words_view])
                print(text_input_view[:20], ' | ', text_input_view[:-20])
                print(pred2words_view[:20], ' | ', pred2words_view[:-20])
                print(ref[:40], ' | ', gen[:40])
                print(" ")
        if (val_loss / len(val_loader)) < best_loss * 0.95:
            now = datetime.datetime.now()
            jobtime = str(now.hour) + str(now.minute)
            modelpath = "snapshots/{}.pt".format(str(jobtime) + "-" + str(epochs))
            torch.save(model.state_dict(), modelpath)
            print("model saved at: ", "snapshots/{}.pt".format(str(jobtime) + "-" + str(epochs)))
            best_loss = current_loss
        print("Validation Loss: {}".format(val_loss / len(val_loader)))


def test(model, test_loader):
    final = []
    with torch.no_grad():
        print("testing...")
        model.eval()
        model.load_state_dict(state_dict=torch.load('snapshots/{}.pt'.format(config.model), map_location=net.device))
        for (batch_num, collate_output) in enumerate(test_loader):
            speech_input, speech_len = collate_output
            speech_input = speech_input.to(device)
            predictions = model(speech_input, speech_len, train=False)
            predictions = predictions.contiguous().view(-1, predictions.size(-1))
            pred2words = torch.argmax(predictions, dim=1)
            # print(pred2words[:].data.detach().cpu().numpy())
            pred2words = [x for x in pred2words if x != 0]
            gen = ''.join([du.letter_list[i - 1] for i in pred2words])
            print(len(gen))
            exit()
            sent = []
            num = 0
            for n, i in enumerate(gen):
                sent.append(i)
                num += 1
                if i == "<eos>" or num ==250:
                    exit()
            final.append(gen)
        # run_write(final)


def main():
    model = net.Seq2Seq(input_dim=40, vocab_size=len(du.vocab), decode_hidden=config.decode_hidden, encode_hidden=config.encode_hidden)
    criterion = nn.CrossEntropyLoss(reduce=None).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if par.train_mode:
        train(model=model, train_loader=du.train_loader, val_loader=du.val_loader, num_epochs=config.num_epochs, criterion=criterion, optimizer=optimizer)
    else:
        test(model=model, test_loader=du.test_loader)


if __name__ == "__main__":
    main()
