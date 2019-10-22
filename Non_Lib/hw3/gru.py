import numpy as np
# import helper.ref_grucell as ref1
import sys

HIDDEN_DIM = 4


class Sigmoid:
    # DO NOT DELETE
    def __init__(self):
        pass
    
    def forward(self, x):
        self.res = 1 / (1 + np.exp(-x))
        return self.res
    
    def backward(self):
        return self.res * (1 - self.res)
    
    def __call__(self, x):
        return self.forward(x)


class Tanh:
    # DO NOT DELETE
    def __init__(self):
        pass
    
    def forward(self, x):
        self.res = np.tanh(x)
        return self.res
    
    def backward(self):
        return 1 - (self.res ** 2)
    
    def __call__(self, x):
        return self.forward(x)


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


class GRU_Cell:
    """docstring for GRU_Cell"""
    
    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0
        
        self.Wzh = np.random.randn(h, h)
        self.Wrh = np.random.randn(h, h)
        self.Wh = np.random.randn(h, h)
        
        self.Wzx = np.random.randn(h, d)
        self.Wrx = np.random.randn(h, d)
        self.Wx = np.random.randn(h, d)
        
        self.dWzh = np.zeros((h, h))
        self.dWrh = np.zeros((h, h))
        self.dWh = np.zeros((h, h))
        
        self.dWzx = np.zeros((h, d))
        self.dWrx = np.zeros((h, d))
        self.dWx = np.zeros((h, d))
        
        self.z_act = Sigmoid()
        self.r_act = Sigmoid()
        self.h_act = Tanh()
        
        # Define other variables to store forward results for backward here
        self.z_t = None
        self.r_t = None
        self.h_tilde_t = None
        self.h_t = None
    
    def init_weights(self, Wzh, Wrh, Wh, Wzx, Wrx, Wx):
        self.Wzh = Wzh
        self.Wrh = Wrh
        self.Wh = Wh
        self.Wzx = Wzx
        self.Wrx = Wrx
        self.Wx = Wx
    
    def __call__(self, x, h):
        return self.forward(x, h)
    
    def forward(self, x, h):
        # input:
        # 	- x: shape(input dim),  observation at current time-step
        # 	- h: shape(hidden dim), hidden-state at previous time-step
        #
        # output:
        # 	- h_t: hidden state at current time-step
        # print(x.shape,h.shape)
        # print(self.Wzh.shape,self.Wzx.shape)
        # # sys.exit(1)
        # x: 3,4    h:3,5
        # Wzh: 4,4  Wzx:4,5

        # self.z_t = self.z_act(np.dot(self.Wzh, h.T) + np.dot(self.Wzx, x.T))
        self.z_t = self.z_act(np.dot(h, self.Wzh.T) + np.dot(x, self.Wzx.T))    # 3,4
        # self.r_t = self.r_act(np.dot(self.Wrh, h.T) + np.dot(self.Wrx, x.T))
        self.r_t = self.r_act(np.dot(h, self.Wrh.T) + np.dot(x, self.Wrx.T))    # 3,4
        assert self.z_t.shape == self.r_t.shape, "different shape between z_t and r_t!"
        self.h_tilde_t = self.h_act(np.dot(np.multiply(self.r_t, h),self.Wh.T) + np.dot(x,self.Wx.T)) # Wh 4,4
        self.h_t = np.multiply((1 - self.z_t), h) + np.multiply(self.z_t, self.h_tilde_t)
        return self.h_t
    
    # raise NotImplementedError
    
    # This  must calculate the gradients wrt the parameters and returns the derivative wrt the inputs, xt and ht, to the cell.
    def backward(self, delta):
        # input:
        #  - delta:  shape (hidden dim), summation of derivative wrt loss from next layer at
        #            the same time-step and derivative wrt loss from same layer at
        #            next time-step
        # output:
        #  - dx: Derivative of loss wrt the input x
        #  - dh: Derivative  of loss wrt the input hidden h
        raise NotImplementedError


# This is the neural net that will run one timestep of the input
# You only need to implement the forward method of this class. 
# This is to test that your GRU Cell implementation is correct when used as a GRU.	
class CharacterPredictor(object):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CharacterPredictor, self).__init__()
    
    # The network consists of a GRU Cell and a linear layer
    
    def init_rnn_weights(self, w_hi, w_hr, w_hn, w_ii, w_ir, w_in):
        # DO NOT MODIFY
        self.rnn.init_weights(w_hi, w_hr, w_hn, w_ii, w_ir, w_in)
    
    def __call__(self, x, h):
        return self.forward(x, h)
    
    def forward(self, x, h):
        # A pass through one time step of the input
        raise NotImplementedError


# An instance of the class defined above runs through a sequence of inputs to generate the logits for all the timesteps.
def inference(net, inputs):
    # input:
    #  - net: An instance of CharacterPredictor
    #  - inputs - a sequence of inputs of dimensions [seq_len x feature_dim]
    # output:
    #  - logits - one per time step of input. Dimensions [seq_len x num_classes]
    raise NotImplementedError


if __name__ == "__main__":
    cell = GRU_Cell(5, HIDDEN_DIM)
    input = np.ones(shape=(6, 3, 5))
    hx = np.ones(shape=(3, 4))
    output = []
    for i in range(6):
        fwd = cell.forward(input[i], hx)
        output.append(fwd)
    # print(output)
# assert fwd == ref1.output
