import sys
import numpy as np
import torch

class dataset(object):
 	"""docstring for ClassName"""
 	def __init__(self):
 		super(dataset, self).__init__()
 		# self.dev_labels = np.load("/Users/robert/Downloads/11-785hw1p2-f19/dev_labels.npy",allow_pickle = True)
 		# self.dev = np.load("/Users/robert/Downloads/11-785hw1p2-f19/dev.npy",allow_pickle=True)
 		self.test = np.load("/Users/robert/Downloads/11-785hw1p2-f19/test.npy",allow_pickle = True)
 		# self.train_lables = np.load("/Users/robert/Downloads/11-785hw1p2-f19/train_labels.npy",allow_pickle=True)
 		# self.train = np.load("/Users/robert/Downloads/11-785hw1p2-f19/train.npy",allow_pickle=True)


f = dataset()
print(f.test)