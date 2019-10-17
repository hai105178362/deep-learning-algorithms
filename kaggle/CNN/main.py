import model_basic as M
import resnet
from cnn_params import num_feats, hidden_sizes, num_classes, learningRate, weightDecay, device, train_dataloader, dev_dataloader,lr_cent,feat_dim
import cnn_params as par
import tracewritter as wrt
import torch

if __name__ == "__main__":
    prev_acc = 0.3
    print(device)
    wrt.log_title(par.allspec)
    # network = M.network
    network = resnet.network
    print("Training...")
    # network.apply(M.init_weights)
    # network.load_state_dict(torch.load('saved_models/17-12-1-e31.pt', map_location=M.device))
    network.train()
    network.to(device)
    # M.train_closs(network, train_dataloader, dev_dataloader)
    resnet.train_closs(network, train_dataloader, dev_dataloader)
