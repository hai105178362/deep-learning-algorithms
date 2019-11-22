import numpy as np
from matplotlib import pyplot as plt
import time
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tests import test_prediction, test_generation
from helper import loader
import csv
from torchnlp.nn import lock_dropout
import torchnlp

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

train_data = np.load('../dataset/wiki.train.npy', allow_pickle=True)
fixtures_pred = np.load('../fixtures/prediction.npz', allow_pickle=True)  # dev
fixtures_gen = np.load('../fixtures/generation.npy', allow_pickle=True)  # dev
fixtures_pred_test = np.load('../fixtures/prediction_test.npz', allow_pickle=True)  # test
fixtures_gen_test = np.load('../fixtures/generation_test.npy', allow_pickle=True)  # test
vocab = np.load('../dataset/vocab.npy', allow_pickle=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dataset = train_data
vocab_size = len(vocab)
BATCH_SIZE = 80
EMBED_SIZE = 400
EMBED_HIDDEN = 1150
HIDDEN_SIZE = 1024
DROP_OUTS = [0.4, 0.3, 0.4, 0.1]
LSTM_LAYERS = 3

vocab_human = []
with open('../dataset/vocab.csv') as f:
    fo = csv.reader(f, delimiter=',')
    vocab_human = np.array([i[1] for i in fo][1:])


class LanguageModelDataLoader(DataLoader):
    """
        TODO: Define data loader logic here
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        data = np.array(dataset)
        if shuffle == True:
            np.random.shuffle(data)
        self.largetext = []
        for i in data:
            self.largetext = np.concatenate((self.largetext, i), axis=None)
        super().__init__(dataset=self.largetext, batch_size=batch_size, shuffle=shuffle)
        self.lenarr = [35, 70]
        self.seqlen = np.random.choice(self.lenarr, 1, p=[0.05, 0.95])
        self.sigma = 5
        # raise NotImplemented

    def __iter__(self):
        start_idx = 0
        tot_len = self.largetext.__len__()
        print("totlen:{}".format(tot_len))
        while True:
            seqlen = int(np.random.normal(self.seqlen, self.sigma))
            if start_idx + seqlen * self.batch_size + 1 >= tot_len:
                break
            sentences = torch.LongTensor(self.largetext[start_idx:start_idx + seqlen * self.batch_size]) \
                .reshape(shape=(self.batch_size, seqlen)).to(DEVICE)
            labels = torch.LongTensor(self.largetext[start_idx + 1:start_idx + seqlen * self.batch_size + 1]) \
                .reshape(shape=(self.batch_size, seqlen)).to(DEVICE)
            start_idx += seqlen * self.batch_size
            yield (sentences, labels)


class LanguageModel(nn.Module):
    """
        TODO: Define your model here
    """

    def __init__(self, vocab_size, hidden=[None, None, None], weight_tie=False):
        super(LanguageModel, self).__init__()
        self.init_hidden = hidden
        self.vocab_size = vocab_size
        self.batch_size = BATCH_SIZE
        self.embed_size = EMBED_SIZE
        self.embed_hidden = EMBED_HIDDEN
        self.hidden_size = HIDDEN_SIZE
        self.lstmlayers = LSTM_LAYERS

        # self.word_lstm_init_h = Variable(torch.randn(self.lstmlayers, self.batch_size, self.embed_hidden).type(torch.FloatTensor), requires_grad=True)
        self.rnns = []
        for l in range(self.lstmlayers):
            if l == 0:
                self.rnns.append(torch.nn.LSTM(self.embed_hidden, self.hidden_size, num_layers=1, dropout=0).to(DEVICE))
            else:
                self.rnns.append(torch.nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1, dropout=0).to(DEVICE))
        self.embedding = torch.nn.Embedding(vocab_size, self.embed_hidden, self.embed_size).to(DEVICE)
        self.rnn = torch.nn.LSTM(input_size=self.embed_hidden, bidirectional=False, hidden_size=self.hidden_size, num_layers=3).to(DEVICE)
        self.scoring = torch.nn.Linear(in_features=self.hidden_size, out_features=vocab_size).to(DEVICE)
        self.drop = torch.nn.Dropout(p=0.1)
        self.locked_dropout1 = torchnlp.nn.LockedDropout(p=DROP_OUTS[0])
        # self.locked_dropout1 = torchnlp.nn.lock_dropout
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.scoring.bias.data.fill_(0)
        self.scoring.weight.data.uniform_(-0.1, 0.1)

    def net_run(self, embed, hidden, validation=False):
        # embed = self.locked_dropout1(embed)
        new_hidden = []
        # raw_output, hidden = self.rnn(emb, hidden)
        cur_outputs = []
        outputs = []
        current_input = embed
        cur_output = None
        for l, rnn in enumerate(self.rnns):
            # print("l:", l)
            cur_output, cur_hidden = rnn(current_input, hidden[l])
            new_hidden.append(cur_hidden)
            cur_outputs.append(cur_output)
            if l != self.lstmlayers - 1:
                cur_output = self.locked_dropout1(cur_output)
                outputs.append(cur_output)
            current_input = cur_output
        hidden = new_hidden
        if validation == True:
            cur_output = cur_output[-1]
        output = self.scoring(cur_output)
        output = self.drop(output)
        outputs.append(output)
        return output, hidden

    def forward(self, x):
        embed = self.embedding(x)
        output, hidden = self.net_run(embed, hidden=self.init_hidden)
        result = output.view(-1, self.batch_size, self.vocab_size)
        return result, hidden

    def predict(self, seq):  # L x V
        embed = self.embedding(seq).unsqueeze(1)
        output, _ = self.net_run(embed, hidden=self.init_hidden, validation=True)
        # _, current_word = torch.max(output, dim=1)  # 1 x 1
        return output

    def generate(self, seq, n_words):  # L x V
        cur_seq = seq
        generated_words = []
        embed = self.embedding(cur_seq).unsqueeze(1)
        output, _ = self.net_run(embed, hidden=self.init_hidden, validation=True)
        _, current_word = torch.max(output, dim=1)  # 1 x 1
        generated_words.append(current_word)
        cur_seq = torch.cat((cur_seq, current_word), dim=0)
        if n_words > 1:
            for i in range(n_words - 1):
                embed = self.embedding(cur_seq).unsqueeze(1)
                output, _ = self.net_run(embed, validation=True, hidden=self.init_hidden)
                _, current_word = torch.max(output, dim=1)  # 1 x 1
                cur_seq = torch.cat((cur_seq, current_word), dim=0)
                generated_words.append(current_word)
                # generated_words = torch.cat((generated_words, current_word),0)
        return torch.cat(generated_words, dim=0)


# model trainer

class LanguageModelTrainer:
    def __init__(self, model, loader, max_epochs=1, run_id='exp'):
        """
            Use this class to train your model
        """
        # feel free to add any other parameters here
        self.model = model
        self.loader = loader
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.predictions_test = []
        self.generated_logits = []
        self.generated = []
        self.generated_logits_test = []
        self.generated_test = []
        self.epochs = 0
        self.max_epochs = max_epochs
        self.run_id = run_id

        # TODO: Define your optimizer and criterion here
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        # self.optimizer = torch.optim.ASGD(model.parameters(), lr=1e-2, weight_decay=1e-7)
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)
        # self.criterion = nn.NLLLoss().to(DEVICE)

    def train(self):
        self.model.train()  # set to training mode
        epoch_loss = 0
        num_batches = 0
        for batch_num, (inputs, targets) in enumerate(self.loader):
            cur_loss = self.train_batch(inputs, targets)
            epoch_loss += cur_loss
            if (batch_num + 1) % 30 == 0:
                print("batch:{}".format(batch_num + 1))
                print("cur_loss is:", cur_loss.item())
        epoch_loss = epoch_loss / (batch_num + 1)
        # epoch_loss.backward()
        # self.optimizer.step()
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f'
              % (self.epochs + 1, self.max_epochs, epoch_loss))
        self.train_losses.append(epoch_loss)

    def train_batch(self, inputs, targets):
        """
            TODO: Define code for training a single batch of inputs

        """
        result, hidden = self.model(inputs)
        loss = self.criterion(result.view(-1, result.size(2)), targets.view(-1))
        # Adding L2 Norm
        par = torch.tensor(10e-5).to(DEVICE)
        l2_reg = torch.tensor(0.).to(DEVICE)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += par * l2_reg
        loss.backward()
        self.optimizer.step()

        return loss

    def test(self):
        # don't change these
        self.model.eval()  # set to eval mode
        predictions = TestLanguageModel.prediction(fixtures_pred['inp'], self.model)  # get predictions
        self.predictions.append(predictions)
        generated_logits = TestLanguageModel.generation(fixtures_gen, 10, self.model)  # generated predictions for 10 words
        generated_logits_test = TestLanguageModel.generation(fixtures_gen_test, 10, self.model)
        nll = test_prediction(predictions, fixtures_pred['out'])
        generated = test_generation(fixtures_gen, generated_logits, vocab)
        generated_test = test_generation(fixtures_gen_test, generated_logits_test, vocab)
        self.val_losses.append(nll)

        self.generated.append(generated)
        self.generated_test.append(generated_test)
        self.generated_logits.append(generated_logits)
        self.generated_logits_test.append(generated_logits_test)

        # generate predictions for test data
        predictions_test = TestLanguageModel.prediction(fixtures_pred_test['inp'], self.model)  # get predictions
        self.predictions_test.append(predictions_test)

        print('[VAL]  Epoch [%d/%d]   Loss: %.4f'
              % (self.epochs + 1, self.max_epochs, nll))
        self.epochs += 1
        return nll

    def save(self):
        # don't change these
        model_path = os.path.join('experiments', self.run_id, 'model-{}.pkl'.format(self.epochs))
        torch.save({'state_dict': self.model.state_dict()},
                   model_path)
        np.save(os.path.join('experiments', self.run_id, 'predictions-{}.npy'.format(self.epochs)), self.predictions[-1])
        np.save(os.path.join('experiments', self.run_id, 'predictions-test-{}.npy'.format(self.epochs)), self.predictions_test[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-{}.npy'.format(self.epochs)), self.generated_logits[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-test-{}.npy'.format(self.epochs)), self.generated_logits_test[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}-test.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated_test[-1])


class TestLanguageModel:
    def prediction(inp, model):
        """
            TODO: write prediction code here

            :param inp:
            :return: a np.ndarray of logits
        """
        print("starting prediction...")
        ans = np.zeros(shape=(1, vocab_size))
        input = torch.LongTensor(inp).to(DEVICE)
        # model.eval()
        for i in input:
            cur_word = model.predict(i).detach().cpu().numpy()
            ans = np.append(ans, cur_word, axis=0)
        return ans[1:]
        raise NotImplemented

    def generation(inp, forward, model):
        """
            TODO: write generation code here

            Generate a sequence of words given a starting sequence.
            :param inp: Initial sequence of words (batch size, length)
            :param forward: number of additional words to generate
            :return: generated words (batch size, forward)
        """
        print("starting generation...")
        # model.eval()
        input = torch.LongTensor(inp).to(DEVICE)
        ans = np.zeros(shape=(1, forward))
        for i in input:
            cur_word = model.generate(i, forward)
            cur_word = cur_word.cpu().detach().numpy()
            ans = np.append(ans, np.array([cur_word]), axis=0)
        return ans[1:].astype(int)
        raise NotImplemented


# TODO: define other hyperparameters here

NUM_EPOCHS = 150
BATCH_SIZE = 80
run_id = str(int(time.time()))
if not os.path.exists('./experiments'):
    os.mkdir('./experiments')
os.mkdir('./experiments/%s' % run_id)
print("Saving models, predictions, and generated words to ./experiments/%s" % run_id)
print("Loader Init...")
loader = LanguageModelDataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Model Init..")
model = LanguageModel(len(vocab))
# model.apply(weights_init)
print("Trainer Init...")
trainer = LanguageModelTrainer(model=model, loader=loader, max_epochs=NUM_EPOCHS, run_id=run_id)
best_nll = 1e30
for epoch in range(NUM_EPOCHS):
    print("Epoch: ", epoch + 1)
    trainer.train()
    nll = trainer.test()
    print("nll: ", nll)
    if nll < best_nll:
        best_nll = nll
        print("Saving model, predictions and generated output for epoch " + str(epoch) + " with NLL: " + str(best_nll))
        trainer.save()
        print("saved")

# Don't change these
# plot training curves
plt.figure()
plt.plot(range(1, trainer.epochs + 1), trainer.train_losses, label='Training losses')
plt.plot(range(1, trainer.epochs + 1), trainer.val_losses, label='Validation losses')
plt.xlabel('Epochs')
plt.ylabel('NLL')
plt.legend()
plt.show()

# see generated output
print(trainer.generated[-1])  # get last generated output
