import torch
import torch.nn as nn
import torch.nn.functional as F

rnn = nn.GRUCell(5, 4)
# input = torch.randn(6, 3, 5)
input = torch.ones(6, 3, 5)
# hx = torch.randn(3, 4)
hx = torch.ones(3, 4)
output = []
for i in range(6):
	hx = rnn(input[i], hx)
	tmp = (hx.detach().numpy())
	output.append(tmp)
print(output[0])