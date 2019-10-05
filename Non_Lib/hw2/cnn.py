from layers import *


# from mlp import *

class CNN_B():
    def __init__(self):
        # Your initialization code goes here
        self.layers = []

    def __call__(self, x):
        f = Flatten()
        print(x.shape)
        x = f(x)
        print(x.shape)
        return self.forward(x)

    def init_weights(self, weights):
        for i in weights:
            self.layers.append(Conv1D(i.shape[0] // 8, i.shape[1], 8, 4))
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
