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
        self.state = np.power((1.0 + np.exp(-x)), -1)
        return self.state
        raise NotImplemented

    def derivative(self):
        # Maybe something we need later in here...
        return self.state * (1.0 - self.state)
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
        curr_max = np.max(self.logits, axis=1)
        self.sm = (np.exp(self.logits) / np.sum(np.exp(self.logits), axis=1).reshape((len(self.logits)), 1))
        logsmtop = np.log(np.exp(self.logits))
        logsmbot = curr_max + np.log(np.sum(np.exp(self.logits - curr_max.reshape((len(self.logits), 1))), axis=1))
        logsm = logsmtop - logsmbot.reshape((len(self.logits), 1))
        return -np.sum(self.labels * logsm, axis=1)
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
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))
        # self.deltagamma = np.zeros((1, fan_in))
        # self.deltabeta = np.zeros((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        if eval:
            return (self.gamma * ((x - self.running_mean) / (np.sqrt(self.running_var) + self.eps))) + self.beta
        r, c = x.shape
        self.mean = np.mean(x, axis=0)
        self.var = np.var(x, axis=0)
        self.norm = (x - self.mean) / (np.sqrt(self.var + self.eps))
        self.out = (self.gamma * self.norm) + self.beta
        # update running batch statistics
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
        self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var
        self.x = x
        return self.out

    def backward(self, delta):
        r, c = delta.shape
        dnorm = (self.gamma * delta)
        self.dbeta = np.sum(delta, axis=0)
        self.dgamma = np.sum(delta * self.norm, axis=0)
        dvar_sqr = -0.5 * np.sum(dnorm * (self.x - self.mean) * np.power((self.var + self.eps), -1.5), axis=0)
        dmiu = -(np.sum(delta * np.power((self.var + self.eps), -0.5), axis=0)) - (
                (2 / r) * self.var * np.sum(self.x - self.mean, axis=0))
        dw = dnorm * np.power((self.var + self.eps), -0.5) + dvar_sqr * (
                2 / r * (self.x - self.mean)) + (dmiu / r)
        return dw
        raise NotImplemented


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    return np.random.normal(np.zeros(shape=(d0, d1)))
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
            for i in hiddens:
                bigArr.append(int(i))
        bigArr.append(output_size)
        self.W = np.array([(weight_init_fn(bigArr[i], bigArr[i + 1])) for i in range(len(bigArr) - 1)])
        self.dW = np.array([(weight_init_fn(bigArr[i], bigArr[i + 1])) for i in range(len(bigArr) - 1)])
        self.b = [(bias_init_fn(bigArr[i + 1])) for i in range(len(bigArr) - 1)]
        self.db = [(bias_init_fn(bigArr[i + 1])) for i in range(len(bigArr) - 1)]
        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = []
        # Feel free to add any other attributes useful to your implementation (input, output, ...)
        self.input = None
        self.state = [np.zeros(shape=(bigArr[i])) for i in range(len(bigArr))]
        self.output = []
        self.batch = []
        self.z = []

        ## Momentum Part
        self.deltaW = [np.zeros(shape=(self.W[i].shape)) for i in range(len(self.W))]
        self.deltaB = [np.zeros(shape=(bigArr[i + 1])) for i in range(len(bigArr) - 1)]
        self.deltagamma = [np.zeros(shape=(bigArr[i + 1])) for i in range(len(bigArr) - 1)]
        self.deltabeta = [np.zeros(shape=(bigArr[i + 1])) for i in range(len(bigArr) - 1)]

    def forward(self, x):
        self.input = np.array(x)
        cur_input = np.array(x)
        bn_layer = self.num_bn_layers
        if len(self.activations) == 1:
            self.state = np.dot(cur_input, self.W[0]) + self.b
            self.output = self.state[0]
            return self.output
        else:
            self.state[0] = x
            for i in range(len(self.activations)):
                dot_product = np.dot(cur_input, self.W[i]) + self.b[i]
                if bn_layer > 0:
                    self.bn_layers.append(BatchNorm(len(dot_product[1])))
                    bn_layer -= 1
                    dot_product = self.bn_layers[bn_layer].forward(dot_product, eval=(self.train_mode != True))
                cur_y = self.activations[i].forward(dot_product)
                self.state[i + 1] = cur_y
                cur_input = cur_y
        self.output = cur_y
        return cur_y
        raise NotImplemented

    def zero_grads(self):
        return np.zeros(shape=(np.shape(self.W)))
        raise NotImplemented

    def step(self):
        for i in range(len(self.W)):
            self.deltaW[i] = self.momentum * self.deltaW[i] - self.lr * self.dW[i]
            self.deltaB[i] = self.momentum * self.deltaB[i] - self.lr * self.db[i]
            self.W[i] += self.deltaW[i]
            self.b[i] += self.deltaB[i]
        for i in range(self.num_bn_layers):
            self.deltagamma[i] = self.bn_layers[i].gamma - self.lr * self.bn_layers[i].dgamma
            self.deltabeta[i] = self.bn_layers[i].beta - self.lr * self.bn_layers[i].dbeta
            self.bn_layers[i].gamma = self.deltagamma[i]
            self.bn_layers[i].beta = self.deltabeta[i]
        return self.W, self.b
        raise NotImplemented

    def backward(self, labels):
        loss = self.criterion(self.output, labels)
        dz_prev = self.criterion.derivative()
        bn_layer = self.num_bn_layers
        if self.nlayers == 1:
            self.dW = np.array([np.matmul(self.input.transpose(), dz_prev)]) / len(self.input)
            self.db = [np.sum(dz_prev, axis=0) / len(self.input)]
            # print("1===")
            # print(self.dW[-1][-1][-1])
            return
        else:
            dz_prev = self.activations[-1].derivative() * dz_prev
            self.dW[-1] = np.matmul(self.state[-2].transpose(), dz_prev) / len(self.input)
            self.db[- 1] = np.sum(dz_prev, axis=0) / len(self.input)
            for i in range(len(self.W) - 1, 0, -1):
                y_prime = np.matmul(dz_prev, self.W[i].transpose())
                if (bn_layer == i and i == 1):
                    cur_z_prime = (y_prime * self.activations[i - 1].derivative())
                    cur_z_prime = self.bn_layers[i - 1].backward(cur_z_prime)
                    bn_layer -= 1
                else:
                    cur_z_prime = (y_prime * self.activations[i - 1].derivative())
                self.dW[i - 1] = np.matmul(self.state[i - 1].transpose(), cur_z_prime) / len(self.input)
                self.db[i - 1] = np.sum(cur_z_prime, axis=0) / len(self.input)
                dz_prev = cur_z_prime
        return
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
    training_losses, training_errors = [], []
    validation_losses, validation_errors = [], []
    l = len(trainx)
    # Setup ...
    for e in range(nepochs):
        print(e)
        accuracy = 0
        np.random.shuffle(idxs)
        train_x = np.array([trainx[idxs[i]] for i in range(len(idxs))])
        train_y = np.array([trainy[idxs[i]] for i in range(len(idxs))])
        train_out = mlp.forward(train_x)
        train_runningloss = SoftmaxCrossEntropy().forward(train_out, train_y)
        mlp.backward(train_y)
        for b in range(0, len(trainx), batch_size):
            mlp.step()
        for i in range(l):
            if np.argmax(train_out[i]) == np.argmax(train_y[i]):
                accuracy += 1
        print("Train_loss: {a:1.5f}  Train_Error: {b:0.5f}  ".format(a=np.mean(train_runningloss), b=(1 - accuracy / l)))
        training_errors.append(1 - (accuracy / l))
        training_losses.append(np.mean(train_runningloss))
        ##########################################
        for b in range(0, len(valx), batch_size):
            accuracy = 0
            mlp.eval()
            eval_out = mlp.forward(valx)
            eval_loss = SoftmaxCrossEntropy().forward(eval_out, valy)
            for i in range(len(valx)):
                if np.argmax(eval_out[i]) == np.argmax(valy[i]):
                    accuracy += 1
        print("Valid_loss: {a:1.5f}  Vlide_Error: {b:0.5f}  ".format(a=np.mean(eval_loss), b=(1 - accuracy / len(valx))))
        print("---------------------------------------------------------------------------------------------------------")
        validation_losses.append(np.mean(eval_loss))
        validation_errors.append(1 - (accuracy / len(valx)))
        ###########################################
        output = []
        for b in range(0, len(testx), batch_size):
            mlp.eval()
            test_out = mlp.forward(testx)
            output.append(test_out.reshape(-1))
    return (training_losses, training_errors, validation_losses, validation_errors)

    raise NotImplemented
