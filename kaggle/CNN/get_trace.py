import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import time
import datetime

all_spec = "NUM_EPOCH:{}   NUM_FEATS:{}   LR:{}   WEIGHT_DECAY:{}\nHIDDEN_SIZE:{}   LR_CENT:{}   FEAT_DIM:{}\n".format(NUM_EPOCHS, NUM_FEATS \
                                                                                                                       , LEARNING_RATE, WEIGHT_DECAY, HIDDEN_SIZE \
                                                                                                                       , LR_CENT, FEAT_DIM)

d = datetime.datetime.today()
record = "{}-{}-{}-e{}".format(d.day, d.hour, d.minute, epoch)
PATH = "saved_models/{}.pt".format(record)
print("*****")
print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
      format(train_loss, train_acc, val_loss, val_acc))
print("*****")
print("==========================================================================\n\n")
cnn_tmplogger = open("cnn_trace.txt", "a")
cnn_tmplogger.write("==========================================================================\n")
cnn_tmplogger.write("Train Accuracy {:.5f} \nValidation Accuracy {:.5f} at Epoch {}\n".format(train_acc, val_acc, epoch + 1))
if train_acc >= 0.4 or val_acc >= 0.4:
    torch.save(model.state_dict(), PATH)
    cnn_tmplogger.write("Model Saved\n")
    print("Model Saved")
cnn_tmplogger.write("==========================================================================\n")
cnn_tmplogger.close()