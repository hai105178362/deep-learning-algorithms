import numpy as np
import math
import sys


class Linear():
    # DO NOT DELETE
    def __init__(self, in_feature, out_feature):
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.W = np.random.randn(out_feature, in_feature)
        self.b = np.zeros(out_feature)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        self.out = x.dot(self.W.T) + self.b
        return self.out

    def backward(self, delta):
        self.db = delta
        self.dW = np.dot(self.x.T, delta)
        dx = np.dot(delta, self.W.T)
        return dx


class Conv1D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        self.W = np.random.randn(out_channel, in_channel, kernel_size)
        self.b = np.zeros(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        self.result_width = 0

    def __call__(self, x):
        # print(self.W.shape,x.shape)
        # self.input = x
        return self.forward(x)

    def forward(self, x):

        ## Your codes here
        self.batch, __, self.width = x.shape
        self.inx = x

        assert __ == self.in_channel, 'Expected the inputs to have {} channels'.format(self.in_channel)
        tmp = self.kernel_size
        while tmp <= self.width:
            if tmp > self.width:
                break
            self.result_width += 1
            tmp += self.stride
        # print("stride: {}, origin witdth: {}, kerneal size: {}, result width: {}".format(self.stride, self.width, self.kernel_size, result_width))
        result = np.zeros(shape=(self.batch, self.out_channel, self.result_width))
        for b in range(self.batch):
            for oc in range(self.out_channel):
                for ic in range(self.in_channel):
                    start, end = 0, self.kernel_size
                    for step in range(self.result_width):
                        result[b][oc][step] += np.sum(np.multiply(self.W[oc][ic][:(end - start)], x[b][ic][start:end])) + self.b[oc]
                        start += self.stride
                        end += self.stride
                        if end > self.width:
                            end = self.width - 1
        return result
        raise NotImplemented

    def backward(self, delta):
        print("delta.shape: {}, kernel_size: {},  W.shape: {}, b.shape: {}".format(delta.shape, self.kernel_size, self.W.shape,self.b.shape))
        dx = np.zeros(shape=(self.batch, self.in_channel, self.width))
        print("dx.shape: {}".format(dx.shape))
        for b in range(self.batch):
            for ic in range(self.in_channel):
                for oc in range(self.out_channel):
                    start, end = 0, self.kernel_size
                    self.db[oc] = np.sum(delta[b][oc])
                    for step in range(self.result_width):
                        dx[b][ic][start:end] += (self.W[b][ic] * delta[b][oc][step])
                        self.dW[b][ic] += self.inx[b][ic][start:end] * delta[b][oc][step]
                        # print(self.inx[b][ic][start:end], delta[b][oc][step])
                        # sys.exit(1)
                        start += self.stride
                        end += self.stride
                        if end > self.width:
                            end = self.width - 1

        ## Your codes here
        # self.db = ???
        # self.dW = ???
        return dx
        raise NotImplemented


class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        ## Your codes here
        raise NotImplemented

    def backward(self, x):
        # Your codes here
        raise NotImplemented


class ReLU():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.dy = (x >= 0).astype(x.dtype)
        return x * self.dy

    def backward(self, delta):
        return self.dy * delta
