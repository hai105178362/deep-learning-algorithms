from layers import *


# from mlp import *

class CNN_B():
    def __init__(self):
        # Your initialization code goes here
        self.layers = []
        self.stride = [4, 1, 1]
        self.kernel_size = [8, 1, 1]
        self.in_channel = [24, 8, 16]
        self.out_channel = [8, 16, 4]
        for i in range(3):
            self.layers.append(Conv1D(self.in_channel[i], self.out_channel[i], self.kernel_size[i], self.stride[i]))
            self.layers.append(ReLU())
        self.layers = self.layers[:-1]  # remove final ReLU
        self.layers.append(Flatten())


    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        for i in range(len(weights)):
            self.layers[i * 2].W = weights[i].T


        return self.layers
        # Load the weights for your CNN from the MLP Weights given
        raise NotImplemented

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta


class CNN_C():
    def __init__(self):
        # Your initialization code goes here
        self.layers = []

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        raise NotImplemented

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
