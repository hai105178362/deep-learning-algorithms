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
    now = datetime.datetime.now()
    job_time = str(now.day) + str(now.hour)
    best_loss = 0.2
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
        epoch_loss = loss_sum / len(train_loader)
        print("Average Training Loss: {}".format(epoch_loss))
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
        if (epoch_loss < best_loss * 0.95) or ((epochs + 1) % 3 == 0 and epochs != 0):
            model_path = "snapshots/{}.pt".format(str(job_time) + "-" + str(epochs))
            torch.save(model.state_dict(), model_path)
            if epoch_loss < best_loss * 0.95:
                best_loss = epoch_loss
            print("best loss: {}".format(best_loss))
            print("model saved at: ", "snapshots/{}.pt".format(str(job_time) + "_" + str(epochs)))


def test(model, test_loader):
    final = []
    with torch.no_grad():
        print("testing...")
        model.eval()
        model.load_state_dict(state_dict=torch.load('snapshots/{}.pt'.format(config.model), map_location=net.device))
        for (batch_num, collate_output) in enumerate(test_loader):
            print("batch: ", batch_num)
            speech_input, speech_len = collate_output
            speech_input = speech_input.to(device)
            predictions = model(speech_input, speech_len, train=False)
            predictions = predictions.contiguous().view(-1, predictions.size(-1))
            pred2words = torch.argmax(predictions, dim=1).detach().cpu().numpy()
            sent = ""
            q = 0
            for i in range(0, len(pred2words), 250):
                tmp = pred2words[i:i + 250]
                if 0 in tmp:
                    pos = np.where(tmp == 0)[0][0]
                    tmp = tmp[:pos]
                    sent = (''.join([du.letter_list[k - 1] for k in tmp]))
                    if "<eos>" in sent:
                        final.append(sent[:sent.index("<eos>")])
                    else:
                        final.append(sent)
                else:
                    sent = (''.join([du.letter_list[k - 1] for k in tmp]))
                    if "<eos>" in sent:
                        final.append(sent[:sent.index("<eos>")])
                    else:
                        final.append(sent)
        run_write(final)


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
