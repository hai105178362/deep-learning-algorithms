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
        return np.ones(self.state.shape)


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
        self.sm = []
        ans = []
        for i in range(len(self.logits)):
            curr_max = np.max(self.logits[i])
            self.sm.append((np.exp(self.logits[i]) / np.sum(np.exp(self.logits[i]))))
            ans.append(-np.sum(self.labels[i] * np.log(self.sm[i])))
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
        # dnrom_dmeanbatch = -pow(self.var+self.eps,-1/2)-0.5(self.x-self.mean)*pow((self.var+self.eps),-3/2)*(-2/len(self.x)*np.sum(self.x-self.mean))
        # dl_dmeanbatch = -dlossbatch_dnorm*pow((self.var+self.eps),1/2)-(1/2))*dlossbatch_dvar*np.sum(self.x-self.mean)
        # dlbatch_dx = dlbatch_d
        # dl_dxhat = self.lossbatch_data * self.gamma
        # dl_dbeta = np.sum(dlossbatch_dy)
        # dl_dgamma = np.sum(dlossbatch_dy * self.norm)
        # dl_dvar = dlossbatch_dnorm * dnorm_dvar
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
        bigArr = [input_size]
        if len(hiddens) > 0:
            # print("hiddens: {}, input: {}, output: {}".format(hiddens, input_size, output_size))
            for i in hiddens:
                bigArr.append(int(i))
        bigArr.append(output_size)
        self.W = [(weight_init_fn(bigArr[i], bigArr[i + 1])) for i in range(len(bigArr) - 1)]
        self.dW = [(weight_init_fn(bigArr[i], bigArr[i + 1])) for i in range(len(bigArr) - 1)]
        self.b = [(bias_init_fn(bigArr[i])) for i in range(len(bigArr) - 1)]
        self.db = [(bias_init_fn(bigArr[i])) for i in range(len(bigArr) - 1)]
        # HINT: self.foo = [ bar(???) for ?? in ? ]

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = None

        # Feel free to add any other attributes useful to your implementation (input, output, ...)
        # *** Add By myself
        self.input = []
        self.state = []
        self.output = []

    def forward(self, x):
        self.input = x
        cur_input = x
        if (len(self.activations) == 1):
            self.state = [np.matmul(cur_input, self.W[0])]
            self.output = self.state[0]
            return self.output
        else:
            self.state.append(cur_input)
            assert len(self.activations) == len(self.W), "Different length between activations and W! {} , {}".format(
                len(self.activations), len(self.W))
            for i in range(len(self.activations)):
                dot_product = np.matmul(cur_input, self.W[i])
                cur_y = self.activations[i].forward(dot_product)
                self.state.append(cur_y)
                cur_input = cur_y
        self.output = self.state[-1]
        return self.output
        raise NotImplemented

    def zero_grads(self):
        return np.zeros(shape=(np.shape(self.W)))
        raise NotImplemented

    def step(self):
        for i in range(len(self.W)):
            self.W[i] += self.lr * self.dW[i]
        return self.W
        raise NotImplemented

    def backward(self, labels):
        self.criterion = SoftmaxCrossEntropy()
        loss = self.criterion(self.output, labels)
        dz_prev = self.criterion.derivative()
        if self.nlayers == 1:
            # print("input:{}  dz_prev:{}".format(self.input.shape, dz_prev.shape))
            self.dW = [np.matmul((self.input).transpose(), dz_prev) / len(self.input)]
            # print(dz_prev.shape)
            self.db = [np.sum(dz_prev, axis=0) / len(self.input)]
            return self.dW
        else:
            # print(self.state[-1].shape, self.state[-2].shape, self.state[0].shape, dz_prev.shape)
            dz_prev = np.multiply(self.activations[-1].derivative(), dz_prev)
            self.dW[-1] = np.matmul(self.state[-2].transpose(), dz_prev) / len(self.state[-1])
            self.db[- 1] = np.sum(dz_prev, axis=0) / len(self.state[-1])
            for i in range(len(self.W) - 1, 0, -1):
                y_prime = np.matmul(dz_prev, self.W[i].transpose())
                cur_z_prime = np.multiply(y_prime, self.activations[i - 1].derivative())
                self.dW[i - 1] = np.matmul(self.state[i - 1].transpose(), cur_z_prime) / len(self.state[i - 1])
                dz_prev = cur_z_prime
                self.db[i - 1] = np.sum(dz_prev, axis=0) / len(self.state[i - 1])
                # print(self.dW[i].shape, self.state[i].shape, dz_prev.shape)
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
