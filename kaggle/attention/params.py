import argparse


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dev_new', type=str, default="dataset.nosync/dev_new.npy")
    parser.add_argument('--path_dev_transcripts', type=str, default="dataset.nosync/dev_transcripts.npy")
    parser.add_argument('--path_test_new', type=str, default="dataset.nosync/dev_new.npy")
    parser.add_argument('--path_train_transcripts', type=str, default="dataset.nosync/dev_transcripts.npy")
    parser.add_argument('--path_train_new', type=str, default="dataset.nosync/dev_transcripts.npy")
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=1500)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=128)
    ### Setting Mode
    parser.add_argument('--mode', type=str, default="dev")
    parser.add_argument('--model', type=str, default=None)
    config = parser.parse_args()
    return config

config = init_parser()
train_mode = True
batch_size = 32
tf_rate = 0.9
if config.mode == "test":
    train_mode = False
