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


# import cnn_params as par


def log_title(allspec):
    d = datetime.datetime.today()
    record = "{}-{}-{}\n".format(d.day, d.hour, d.minute)
    cnn_logger = open("tracelog.txt", "a")
    cnn_logger.write("==========================================================================\n")
    cnn_logger.write(record)
    cnn_logger.write(allspec + "\n\n")
    cnn_logger.close()


def recordtrace(train_acc, train_loss, val_acc, val_loss, epoch):
        cnn_logger = open("tracelog.txt", "a")
        cnn_logger.write('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f} \t epoch: {}\n'.format(train_loss, train_acc, val_loss, val_acc, epoch))
        cnn_logger.close()
