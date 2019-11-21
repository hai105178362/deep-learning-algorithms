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
batch_size = 80
embed_size = 400
embed_hidden = 1150
hidden_size = 256


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


vocab_human = []
with open('../dataset/vocab.csv') as f:
    fo = csv.reader(f, delimiter=',')
    vocab_human = np.array([i[1] for i in fo][1:])


class LanguageModel(nn.Module):
    """
        TODO: Define your model here
    """

    def __init__(self, vocab_size):
        super(LanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.embed_hidden = embed_hidden
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, self.embed_hidden, self.embed_size).to(DEVICE)
        self.rnn = torch.nn.LSTM(input_size=self.embed_hidden, hidden_size=self.hidden_size, num_layers=1).to(DEVICE)
        self.scoring = torch.nn.Linear(in_features=self.hidden_size, out_features=vocab_size).to(DEVICE)

    def forward(self, x):
        # print("Embedding...")
        result = self.embedding(x)
        output, hidden = self.rnn(result)
        # print("flattening..")
        output_lstm_flatten = output.view(-1, self.hidden_size)
        output_flatten = self.scoring(output_lstm_flatten)
        # return output_flatten
        return output_flatten.view(-1, self.batch_size, self.vocab_size)
        # Feel free to add extra arguments to forward (like an argument to pass in the hiddens)
        raise NotImplemented

    def generate(self, seq, n_words):  # L x V
        # performs greedy search to extract and return words (one sequence).
        generated_words = []
        embed = self.embedding(seq).unsqueeze(1)  # L x 1 x E
        hidden = None
        output_lstm, hidden = self.rnn(embed, hidden)  # L x 1 x H
        output = output_lstm[-1]  # 1 x H
        scores = self.scoring(output)  # 1 x V
        # _, current_word = torch.max(scores, dim=1)  # 1 x 1
        _, current_word = torch.max(scores, dim=1)  # 1 x 1
        # generated_words.append(current_word)
        generated_words.append(scores)
        if n_words > 1:
            for i in range(n_words - 1):
                embed = self.embedding(current_word).unsqueeze(0)  # 1 x 1 x E
                output_lstm, hidden = self.rnn(embed, hidden)  # 1 x 1 x H
                output = output_lstm[0]  # 1 x H
                scores = self.scoring(output)  # V
                # _, current_word = torch.max(scores, dim=1)  # 1
                # generated_words.append(current_word)
                generated_words.append(scores)
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
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-7)
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)
        # self.criterion = nn.NLLLoss()

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
        self.epochs += 1

    def train_batch(self, inputs, targets):
        """
            TODO: Define code for training a single batch of inputs

        """
        # print(targets.view(-1))
        # with torch.no_grad():
        #     result = self.model(inputs)
        #     flat = result.view(-1, result.size(2))
        #     print(flat)
        #     out = np.argmax(flat,axis=1)
        #     print(vocab_human[out[-1]])
        # exit()
        # print(inputs.shape)
        result = self.model(inputs)
        loss = self.criterion(result.view(-1, result.size(2)), targets.view(-1))
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
        ans = []
        input = torch.LongTensor(inp).to(DEVICE)
        with torch.no_grad():
            for i in input:
                cur_word = model.generate(i, 1).cpu().numpy()
                ans = np.append(ans, cur_word.ravel(), axis=0)
                # print("cur_word:",cur_word.shape)
        print("ans: ", ans)
        print(len(ans))
        return ans
        #     print("generating input...")
        #     input = torch.LongTensor(inp).to(DEVICE)
        #     print("input size:{}".format(input.shape))
        #     print("getting result...")
        #     result = model(input)
        #     print("model result:", result.shape)
        #     print("flattening...")
        #     flat = result.view(-1, result.size(2))
        #     print("flat:", flat.shape)
        #     out = (torch.argmax(flat, axis=1))
        #     print("out", out.shape)
        #     print(out)
        #     # ans.append(out[-1])
        #     # print("Prediction:{}".format(vocab_human[out[-1]]))
        # print("========PREDICTION=======")
        # # print(ans)
        # return out
        # # return ans
        raise NotImplemented

    def generation(inp, forward, model):
        """
            TODO: write generation code here

            Generate a sequence of words given a starting sequence.
            :param inp: Initial sequence of words (batch size, length)
            :param forward: number of additional words to generate
            :return: generated words (batch size, forward)
        """
        embedding = torch.nn.Embedding(vocab_size, embed_hidden, embed_size).to(DEVICE)
        rnn = torch.nn.LSTM(input_size=embed_hidden, hidden_size=hidden_size, num_layers=1).to(DEVICE)
        linear = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size).to(DEVICE)

        with torch.no_grad():
            input = torch.LongTensor(inp).to(DEVICE)
            ans = []
            for i in input:
                cur_word = model.generate(i, forward)
                # print(cur_word)
                # print(cur_word.data)
                cur_word = (cur_word.cpu().numpy())[0]
                # exit()
                ans.append(cur_word)
            print(ans)
            print(len(ans))
            return ans
            raise NotImplemented


# TODO: define other hyperparameters here

NUM_EPOCHS = 2
BATCH_SIZE = 80
run_id = str(int(time.time()))
# if not os.path.exists('./experiments'):
#     os.mkdir('./experiments')
# os.mkdir('./experiments/%s' % run_id)
# print("Saving models, predictions, and generated words to ./experiments/%s" % run_id)
# print("Loader Init...")
loader = LanguageModelDataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Model Init..")
model = LanguageModel(len(vocab))
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
