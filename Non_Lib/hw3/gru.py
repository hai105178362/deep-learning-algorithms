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
        
        # self.Wzh = np.ones((h, h))
        # self.Wrh = np.ones((h, h))
        # self.Wh = np.ones((h, h))
        #
        # self.Wzx = np.ones((h, d))
        # self.Wrx = np.ones((h, d))
        # self.Wx = np.ones((h, d))
        
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
        
        self.h = np.array([h])
        self.x = np.array([x])
        # self.z_t = self.z_act(np.dot(self.Wzh, h.T) + np.dot(self.Wzx, x.T))
        self.z_t = self.z_act(np.dot(self.h, self.Wzh.T) + np.dot(self.x, self.Wzx.T))
        # self.r_t = self.r_act(np.dot(self.Wrh, h.T) + np.dot(self.Wrx, x.T))
        self.r_t = self.r_act(np.dot(self.h, self.Wrh.T) + np.dot(self.x, self.Wrx.T))
        assert self.z_t.shape == self.r_t.shape, "different shape between z_t and r_t!"
        self.h_tilde_t = self.h_act(np.dot(np.multiply(self.r_t, self.h), self.Wh.T) + np.dot(self.x, self.Wx.T))  # Wh 4,4
        self.h_t = np.multiply((1 - self.z_t), self.h) + np.multiply(self.z_t, self.h_tilde_t)
        
        # print("self.z_t:{}   self.r_t:{}".format(self.z_t.shape, self.r_t.shape))
        # print("self.h_tilde_t:{}   self.h_t:{}".format(self.h_tilde_t.shape, self.h_t.shape))
        # print("==========FORWARD END=========")
        return self.h_t[0]
    
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
        #########
        print("X:{}  H:{} ".format(self.x, self.h))
        print("delta: {}".format(delta))
        print("Wrx:{}\nWzx:{}\nWx:{}\nWrh:{}\nWzh:{}\nWh:{}".format(self.Wrx, self.Wzx, self.Wx, self.Wrh, self.Wzh, self.Wh))
        # print(self.h)
        # print(self.x)
        
        # Part 1
        dl_dz = np.multiply(delta, -1 * (self.h)) + np.multiply(delta, self.h_tilde_t)
        assert dl_dz.shape == self.z_t.shape
        dl_dzx, dl_dzh = np.matmul(dl_dz * self.z_act.backward(), self.Wzx), np.matmul(dl_dz * self.z_act.backward(), self.Wzh)
        assert dl_dzx.shape == self.x.shape and dl_dzh.shape == self.h.shape
        
        # Part 2
        dl_dhtilde = np.multiply(delta, self.z_t)
        assert dl_dhtilde.shape == self.h_tilde_t.shape
        dl_dr = np.matmul(dl_dhtilde * self.h_act.backward(), self.Wh) * self.h
        dl_drx, dl_drh = np.matmul(dl_dr * self.r_act.backward(), self.Wrx), np.matmul(dl_dr * self.r_act.backward(), self.Wrh)
        assert dl_drx.shape == self.x.shape and dl_drh.shape == self.h.shape
        
        # Part 3
        # print(self.h_act.backward().shape,self.Wx.shape)
        dl_dhtilde_x, dl_dhtilde_h = np.matmul(dl_dhtilde * self.h_act.backward(), self.Wx), np.matmul(dl_dhtilde * self.h_act.backward(), self.Wh) * self.r_t
        assert dl_dhtilde_x.shape == self.x.shape and dl_dhtilde_h.shape == self.h.shape
        dx = dl_dzx + dl_drx + dl_dhtilde_x
        dh = dl_dzh + dl_drh + dl_dhtilde_h + (1 - self.z_t) * delta
        print("dx: {}".format(dx))
        print("dh: {}".format(dh))
        
        ## Weight update
        print((dl_dr * self.r_act.backward()).shape,self.x.shape)
        self.dWrx, self.dWrh = np.matmul((dl_dr * self.r_act.backward()).T, self.x), np.matmul((dl_dr * self.r_act.backward()).T, self.h)
        self.dWzx, self.dWzh = np.matmul((dl_dz * self.z_act.backward()).T, self.x), np.matmul((dl_dz * self.z_act.backward()).T, self.h)
        self.dWx, self.dWh = np.matmul((dl_dhtilde * self.h_act.backward()).T, self.x), np.matmul((dl_dhtilde * self.h_act.backward()).T, self.h) * self.r_t
        return dx, dh
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
    # cell = GRU_Cell(5, HIDDEN_DIM)
    cell = GRU_Cell(5, 2)
    # input = np.ones(shape=(5,))
    x = [1.62434536, -0.61175641, -0.52817175, -1.07296862, 0.86540763]
    # x = [1, 1, 3, 4, 5]
    # h = [1, 2]
    h = [0.30017032, -0.35224985]
    cell.Wrx = np.array([[0.31563495, -2.02220122, -0.30620401, 0.82797464, 0.23009474],
                         [0.76201118, -0.22232814, -0.20075807, 0.18656139, 0.41005165]])
    cell.Wzx = np.array([[0.48851815, -0.07557171, 1.13162939, 1.51981682, 2.18557541],
                         [-1.39649634, -1.44411381, -0.50446586, 0.16003707, 0.87616892]])
    cell.Wx = np.array([[0.19829972, 0.11900865, -0.67066229, 0.37756379, 0.12182127],
                        [1.12948391, 1.19891788, 0.18515642, -0.37528495, -0.63873041]])
    cell.Wrh = np.array([[0.83898341, 0.93110208],
                         [0.28558733, 0.88514116]])
    cell.Wzh = np.array([[-1.1425182, -0.34934272],
                         [-0.20889423, 0.58662319]])
    cell.Wh = np.array([[-0.75439794, 1.25286816],
                        [0.51292982, -0.29809284]])
    # hx = np.ones(shape=(1, 2))
    # hx = np.ones(shape=(2,))
    # output = []
    # output = np.array(output)
    fwd = cell.forward(x, h)
    # print(output.shape)
    # delta = np.array([0.52057634, -1.14434139])
    
    delta = [0.52057634, -1.14434139]
    # delta = [1, 2]
    # print(delta)
    ans = cell.backward(delta)
    print("Refx:", [0.23557079, 0.31226805, -0.14046534, -0.00406543, -0.2789988])
