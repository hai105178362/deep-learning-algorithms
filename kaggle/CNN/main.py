import cnnmodel as M
from cnn_params import num_feats, hidden_sizes, num_classes, learningRate, weightDecay, device, train_dataloader, dev_dataloader,lr_cent,feat_dim
import cnn_params as par
import tracewritter as wrt

if __name__ == "__main__":
    print(device)
    print("Starting CNN")
    wrt.log_title(par.allspec)
    network = M.network
    network.apply(M.init_weights)
    network.train()
    network.to(device)
    M.train_closs(network, train_dataloader, dev_dataloader)
