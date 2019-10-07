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
        self.layers[0].W = np.zeros(shape=(self.out_channel[0], self.in_channel[0], self.kernel_size[0]))  # (out_channel,in_channel,kernel_size)
        self.layers[2].W = np.zeros(shape=(self.out_channel[1], self.in_channel[1], self.kernel_size[1]))
        self.layers[4].W = np.zeros(shape=(self.out_channel[2], self.in_channel[2], self.kernel_size[2]))

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        for k in range(self.out_channel[0]):
            cnt = 0
            for j in range(self.kernel_size[0]):
                for i in range(self.in_channel[0]):
                    self.layers[0].W[k][i][j] = weights[0][cnt][k]
                    cnt += 1

        for k in range(self.in_channel[1]):
            for j in range(self.out_channel[1]):
                for i in range(self.kernel_size[1]):
                    self.layers[2].W[j][k][i] = weights[1][k][j]
        #
        for k in range(self.in_channel[2]):
            for j in range(self.out_channel[2]):
                for i in range(self.kernel_size[2]):
                    self.layers[4].W[j][k][i] = weights[2][k][j]
        return self.layers

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
        self.stride = [2, 2, 1]
        self.kernel_size = [2, 2, 2]
        self.in_channel = [24, 2, 8]
        self.out_channel = [2, 8, 4]
        for i in range(3):
            self.layers.append(Conv1D(self.in_channel[i], self.out_channel[i], self.kernel_size[i], self.stride[i]))
            self.layers.append(ReLU())
        self.layers = self.layers[:-1]  # remove final ReLU
        self.layers.append(Flatten())

        self.layers[0].W = np.zeros(shape=(self.out_channel[0], self.in_channel[0], self.kernel_size[0]))  # (out_channel,in_channel,kernel_size)
        self.layers[2].W = np.zeros(shape=(self.out_channel[1], self.in_channel[1], self.kernel_size[1]))
        self.layers[4].W = np.zeros(shape=(self.out_channel[2], self.in_channel[2], self.kernel_size[2]))

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        new_wegiht = []
        for i in range(len(weights)):
            tmp = []
            for j in weights[i]:
                tmp.append(j[~np.isnan(j)])
            new_wegiht.append(np.array(tmp))
        ##########################################

        for k in range(self.out_channel[0]):  # 2
            cnt = 0
            for j in range(self.kernel_size[0]):  # 2
                for i in range(self.in_channel[0]):  # 24
                    self.layers[0].W[k][i][j] = new_wegiht[0][cnt][k]
                    cnt += 1

        for k in range(self.out_channel[1]):  # 8
            cnt = 0
            for i in range(self.kernel_size[1]):  # 2
                for j in range(self.in_channel[1]):  # 2
                    self.layers[2].W[k][j][i] = new_wegiht[1][cnt][k]
                    cnt += 1

        for k in range(self.out_channel[2]):  # 4
            cnt = 0
            for i in range(self.kernel_size[2]):  # 2
                for j in range(self.in_channel[2]):  # 8
                    self.layers[4].W[k][j][i] = new_wegiht[2][cnt][k]
                    cnt += 1

        return self.layers

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        # print(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
