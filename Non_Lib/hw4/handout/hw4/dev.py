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


# for i in train_data:
#     dataset = np.concatenate((dataset,i),axis=None)
#     # exit()
# print((dataset).shape)
# print(np.random.normal(100, 0.1))
# exit()

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
        num_iters = self.largetext.__len__() // self.batch_size
        if (self.largetext.__len__() % self.batch_size) != 0: num_iters += 1
        print("num_iters: {}".format(num_iters))
        idx = 0
        randnum = int(np.random.normal(self.seqlen, self.sigma))
        for i in range(num_iters + 1):
            sentences = np.empty(shape=(1, randnum))
            labels = np.empty(shape=(1, randnum))
            for j in range(self.batch_size):
                if i * self.batch_size + j + randnum + 1 < len(self.largetext):
                    idx = i * self.batch_size + j
                    cur_sentence = self.largetext[idx: idx + randnum]
                    cur_label = self.largetext[idx + 1:idx + randnum + 1]
                    sentences = np.append(sentences, np.array([cur_sentence]), axis=0)
                    labels = np.append(labels, np.array([cur_label]), axis=0)
                # print(sentences)
            yield (torch.LongTensor(sentences[1:]), torch.LongTensor(labels[1:]))


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
        self.embed = torch.nn.Embedding(vocab_size, 1150, 400).to(DEVICE)
        self.lstm = torch.nn.LSTM(1150, 1150, bidirectional=False, num_layers=1).to(DEVICE)
        self.linear = torch.nn.Linear(in_features=1150, out_features=vocab_size).to(DEVICE)

        # raise NotImplemented

    def forward(self, x):
        print("Embedding...")
        result = self.embed(x)
        cur_shape = result.shape
        result = result.reshape(cur_shape[1], cur_shape[0], cur_shape[2])
        print(result.shape)
        print("Running LSTM...")
        result = self.lstm(result)[0]
        # print("LSTM result: {}".format(result))
        print("Linear Layer...")
        result = self.linear(result)
        print("Reult is: {}".format(result.shape))
        return result
        # Feel free to add extra arguments to forward (like an argument to pass in the hiddens)
        raise NotImplemented


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
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.NLLLoss()

    def train(self):
        self.model.train()  # set to training mode
        epoch_loss = 0
        num_batches = 0
        for batch_num, (inputs, targets) in enumerate(self.loader):
            epoch_loss += self.train_batch(inputs, targets)
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f'
              % (self.epochs + 1, self.max_epochs, epoch_loss))
        self.train_losses.append(epoch_loss)

    def train_batch(self, inputs, targets):
        """
            TODO: Define code for training a single batch of inputs

        """
        # input_shape = inputs.shape
        # mask = np.zeros(shape=(input_shape[0],len(vocab)))

        loss = 0
        cur_result = self.model(inputs)
        cur_shape = cur_result.shape
        # print(cur_result)
        print("input shape: ", cur_result.shape, "target shape:", targets.shape)
        new_result = torch.argmax(cur_result,dim=2).T
        print(new_result.shape)
        curr_loss = self.criterion(new_result, targets)
        print("batch loss:", loss)
        loss += curr_loss
        curr_loss.backward()
        self.optimizer.step()
        return loss
        raise NotImplemented

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
        out = model(inp, len(vocab))
        return out
        raise NotImplemented

    def generation(inp, forward, model):
        """
            TODO: write generation code here

            Generate a sequence of words given a starting sequence.
            :param inp: Initial sequence of words (batch size, length)
            :param forward: number of additional words to generate
            :return: generated words (batch size, forward)
        """
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
# for batch_num, (inputs, targets) in enumerate(loader):
#     print(inputs.shape)
#     # tmp = inputs .unsqueeze(dim=0)
#     # print(tmp.shape)
#     print("============")
#     print(targets.shape)
#     exit(1)

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
