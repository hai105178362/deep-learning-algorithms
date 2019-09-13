"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os

import sys


class Activation(object):
    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):
    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):
    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed to stay the same for AL

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.state = 1.0 / (1 + np.exp(-x))
        return self.state
        raise NotImplemented

    def derivative(self):
        # Maybe something we need later in here...
        return self.state * (1 - self.state)
        raise NotImplemented


class Tanh(Activation):
    """
    Tanh non-linearity
    """

    # This one's all you!

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.state

        raise NotImplemented

    def derivative(self):
        return 1 - np.power(self.state, 2)
        raise NotImplemented


class ReLU(Activation):
    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = np.where(x <= 0, 0, x)
        return self.state
        raise NotImplemented

    def derivative(self):
        return np.where(self.state > 0, 1.0, 0.0)
        raise NotImplemented


# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):

        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):
        self.logits = x
        self.labels = y
        self.sm = np.zeros(shape=(len(self.logits), len(self.logits[0])))
        ans = np.zeros(shape=(len(self.logits),))
        rowlen = len(self.logits[0])
        for i in range(len(self.logits)):
            curr_max = np.max(self.logits[i])
            for j in range(rowlen):
                self.sm[i][j] = (np.exp(self.logits[i][j]) / np.sum(np.exp(self.logits[i])))
            ans[i] = -np.sum(self.labels[i] * np.log(self.sm[i]))
        return np.array(ans)
        # ...

        raise NotImplemented

    def derivative(self):
        # self.sm might be useful here...
        return self.sm - self.labels
        raise NotImplemented


class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):
        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = []
        self.mean = []

        self.gamma = []
        self.dgamma = []

        self.beta = []
        self.dbeta = []

        # inference parameters
        self.running_mean = []
        self.running_var = []

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        # if eval:
        #    # ???
        self.x = x
        self.norm = np.zeros(shape=np.shape(x))
        for i in range(len(self.x)):
            self.var.append(np.var(self.x[i]))
            self.mean.append(np.mean(self.x[i]))
        for i in range(len(self.x)):
            for j in range(len(self.x[i])):
                self.norm[i][j] = (x[i][j] - self.mean[0][i]) / (np.sqrt(self.var[i][j] + self.eps))
        self.out = self.gamma * self.norm + self.beta
        # self.mean = # ???
        # self.var = # ???
        # self.norm = # ???
        # self.out = # ???

        # update running batch statistics
        # self.running_mean = # ???
        # self.running_var = # ???

        # ...
        return self.out

    def backward(self, delta):
        return self.out
        raise NotImplemented


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    return np.randomn(shape=(d0, d1))
    raise NotImplemented


def zeros_bias_init(d):
    return np.zeros(shape=(1, d))
    raise NotImplemented


class MLP(object):
    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr,
                 momentum=0.0, num_bn_layers=0):
        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        self.W = None
        self.dW = None
        self.dW = []
        self.b = None
        self.db = None
        # HINT: self.foo = [ bar(???) for ?? in ? ]

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = None

        # Feel free to add any other attributes useful to your implementation (input, output, ...)
        # *** Add By myself
        self.input = []
        self.z = []
        self.yhat = []
        self.output = []
        self.W = []
        self.cur_gradient = []
        self.dW = []
        self.tmpout = []
        self.backdrv = []

        # *** Generate Weight Matrix
        if (len(hiddens) > 0):
            self.W.append(weight_init_fn(input_size, hiddens[0]))
            for i in range(len(hiddens) - 1):
                self.W.append(weight_init_fn(hiddens[i], hiddens[i + 1]))
            self.W.append(weight_init_fn(hiddens[-1], output_size))
            self.b = bias_init_fn(len(hiddens) + 1)
        else:
            self.W.append(weight_init_fn(input_size, output_size))
            self.b = bias_init_fn(len(hiddens) + 1)

    def forward(self, x):
        self.input = x
        self.output = np.zeros(shape=(len(x),self.output_size))
        # print(len(x[0]))
        l = len(self.W)
        curr_y = x
        for i in range(len(x)):
            tmp = []
            for j in range(l):
                if j == 0:
                    dot_product = np.dot(curr_y[i], self.W[j]) + self.b[0][j]
                    curr_result = self.activations[j].forward(dot_product + self.b[0][j])
                else:
                    dot_product = np.dot(curr_result, self.W[j]) + self.b[0][j]
                    curr_result = self.activations[j].forward(dot_product + self.b[0][j])
                tmp = curr_result
            self.output[i] = tmp
        return self.output
        raise NotImplemented

    def zero_grads(self):
        return  np.zeros(shape=(np.shape(self.W)))
        raise NotImplemented

    def step(self):
        return self.backdrv
        raise NotImplemented

    def backward(self, labels):
        self.dW = np.zeros(shape=np.shape(self.W))
        self.criterion = SoftmaxCrossEntropy()
        self.yhat = self.yhat
        loss = self.criterion.forward(self.output[-1], labels)
        loss_backward = self.criterion.derivative()
        prev_gradient = loss_backward
        # print(np.shape(self.input))
        # for i in range((len(self.activations)-1), -1, -1):
        #     self.cur_gradient = np.dot(self.W[i],prev_gradient.T)
        #     if i<(len(self.activations)-1):
        #         prev_gradient = np.dot(self.W[i],self.activations[i].derivative().T)
        #     print("1:{} 2:{} 3:{}".format(np.shape(prev_gradient), np.shape(self.W[i]),np.shape(self.activations[i].derivative())))
        #     self.dW.append(np.dot(self.yhat[i].T, self.cur_gradient))
        #
        # if len(self.activations) > 1:
        #     self.dW = np.flipud(self.dW)
        for row in range(len(self.input)):
            for i in range((len(self.activations)-1), -1, -1):
                self.cur_gradient = np.dot(self.W[i],prev_gradient.T)
                if i<(len(self.activations)-1):
                    prev_gradient = np.dot(self.W[i],self.activations[i].derivative().T)
                # print(np.shape(self.yhat[row]),np.shape(self.cur_gradient))
                self.dW[row] += (np.multiply(self.yhat[row][i].T, self.cur_gradient))
            if len(self.activations) > 1:
                self.dW[row] = np.flipud(self.dW)

        return self.dW
        raise NotImplemented


def __call__(self, x):
    return self.forward(x)


def train(self):
    self.train_mode = True


def eval(self):
    self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):
    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    # Setup ...

    for e in range(nepochs):

        # Per epoch setup ...

        for b in range(0, len(trainx), batch_size):
            pass  # Remove this line when you start implementing this
            # Train ...

        for b in range(0, len(valx), batch_size):
            pass  # Remove this line when you start implementing this
            # Val ...

        # Accumulate data...

    # Cleanup ...

    for b in range(0, len(testx), batch_size):
        pass  # Remove this line when you start implementing this
        # Test ...

    # Return results ...

    # return (training_losses, training_errors, validation_losses, validation_errors)

    raise NotImplemented
