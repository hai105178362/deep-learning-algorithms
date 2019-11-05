import torch
from ctcdecode import CTCBeamDecoder

decoder = CTCBeamDecoder([' ', 'A'], beam_width=4)
probs = torch.Tensor([[0.2, 0.8], [0.8, 0.2]]).unsqueeze(0)
print(probs.size())
out, _, _, out_lens = decoder.decode(probs, torch.LongTensor([2]))
print(out[0, 0, :out_lens[0, 0]])
print(out)