import argparse


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dev_new', type=str, default="dataset.nosync/dev_new.npy")
    parser.add_argument('--path_dev_transcripts', type=str, default="dataset.nosync/dev_transcripts.npy")
    parser.add_argument('--path_test_new', type=str, default="dataset.nosync/dev_new.npy")
    parser.add_argument('--path_train_transcripts', type=str, default="dataset.nosync/train_transcripts.npy")
    parser.add_argument('--path_train_new', type=str, default="dataset.nosync/train_new.npy")
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--num_epochs', type=int, default=1500)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--num_workers', type=int, default=1)
    ### Hidden Size
    parser.add_argument('--encode_hidden', type=int, default=256)
    parser.add_argument('--decode_hidden', type=int, default=512)
    parser.add_argument('--attention_hidden', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=256)

    # parser.add_argument('--encode_hidden', type=int, default=32)
    # parser.add_argument('--decode_hidden', type=int, default=32)
    # parser.add_argument('--attention_hidden', type=int, default=32)
    # parser.add_argument('--embed_dim', type=int, default=32)

    ### Setting Mode
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--model', type=str, default=None)
    config = parser.parse_args()
    return config

config = init_parser()
train_mode = True
tf_rate = 0.9
if config.mode == "test":
    train_mode = False
