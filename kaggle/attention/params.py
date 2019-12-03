import argparse


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dev_new', type=str, default="dataset.nosync/dev_new.npy")
    parser.add_argument('--path_dev_transcripts', type=str, default="dataset.nosync/dev_transcripts.npy")
    parser.add_argument('--path_test_new', type=str, default="dataset.nosync/dev_transcripts.npy")
    parser.add_argument('--path_train_transcripts', type=str, default="dataset.nosync/dev_transcripts.npy")
    parser.add_argument('--path_train_new', type=str, default="dataset.nosync/dev_transcripts.npy")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=1500)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default="dev")
    parser.add_argument('--hidden_dim', type=int, default=128)
    config = parser.parse_args()
    return config


config = init_parser()
