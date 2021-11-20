import numpy as np

## by yourself .Finish your own NN framework
## Just an example.You can alter sample code anywhere. 

class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *input):
        r"""Define the forward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def backward(self, *output_grad):
        r"""Define the backward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError
        
## by yourself .Finish your own NN framework
class FullyConnected(_Layer):
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((1, out_features))
        self.input = np.array([])

        self.weight_grad = np.empty([])
        self.bias_grad = np.empty([])

    def forward(self, input):
        output = input.dot(self.weight) + self.bias
        self.input = input
        return output

    def backward(self, output_grad):
        input_grad = output_grad.dot(self.weight.T)
        self.weight_grad = self.input.T.dot(output_grad)
        self.bias_grad = output_grad.mean(axis = 0)
        
        return input_grad
    

## by yourself .Finish your own NN framework
class Leaky_ReLU(_Layer): 
    def __init__(self):
        self.input = np.empty([])

    def forward(self, input): 
        self.input = input
        output = input.copy()
        output[self.input<0] *= 0.01
        return output

    def backward(self, input_grad):
        output_grad = input_grad.copy()
        output_grad[self.input<0] *= 0.01
        
        return output_grad

class SoftmaxWithloss(_Layer):
    def __init__(self):
        self.predict = np.empty([])

    def forward(self, input, target):
        self.predict = input
        for i in range(len(input)):
            app = np.exp(input[i]-np.max(input[i]))
            self.predict[i] = app / np.sum(app)
            
        col_num = target.shape[0]

        log_likelihood = -np.log(self.predict)*(target)
        your_loss = np.sum(log_likelihood) / col_num

        return self.predict, your_loss

    def backward(self, target):
        input_grad = self.predict - target
        return input_grad
    
class Adam(_Layer):
    def __init__(self,b1 = 0.9, b2 = 0.999, eps = 1e-8):
        self.m_wgrad, self.v_wgrad = 0, 0
        self.m_bgrad, self.v_bgrad = 0, 0
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 1
    
    def update(self, fc_obj, lr):
        self.m_wgrad = self.b1*self.m_wgrad + (1-self.b1)*fc_obj.weight_grad
        self.m_bgrad = self.b1*self.m_bgrad + (1-self.b1)*fc_obj.bias_grad
        
        self.v_wgrad = self.b2*self.v_wgrad + (1-self.b2)*(fc_obj.weight_grad**2)
        self.v_bgrad = self.b2*self.v_bgrad + (1-self.b2)*(fc_obj.bias_grad**2)
        
        m_wgrad_corr = self.m_wgrad / (1 - (self.b1**self.t))
        m_bgrad_corr = self.m_bgrad / (1 - (self.b1**self.t))
        v_wgrad_corr = self.v_wgrad / (1 - (self.b2**self.t))
        v_bgrad_corr = self.v_bgrad / (1 - (self.b2**self.t))
        
        #mutable -> call by reference
        fc_obj.weight -= lr*(m_wgrad_corr/(np.sqrt(v_wgrad_corr)+self.eps))
        fc_obj.bias -= lr*(m_bgrad_corr/(np.sqrt(v_bgrad_corr)+self.eps))
        
        self.t += 1
        
## CNN

def get_indices(X_shape, HF, WF, stride, pad):

    m, n_C, n_H, n_W = X_shape

    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1


    level1 = np.repeat(np.arange(HF), WF)
    level1 = np.tile(level1, n_C)
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    
    slide1 = np.tile(np.arange(WF), HF)
    slide1 = np.tile(slide1, n_C)
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d

def im2col(X, HF, WF, stride, pad):
    
    X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols


def col2im(dX_col, X_shape, HF, WF, stride, pad):
    
    
    N, D, H, W = X_shape
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((N, D, H_padded, W_padded))
    
    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))
    np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[pad:-pad, pad:-pad, :, :]
    
class Conv():
    
    def __init__(self, nb_filters, filter_size, nb_channels, stride=1, padding=0):
        self.n_F = nb_filters
        self.f = filter_size
        self.n_C = nb_channels
        self.s = stride
        self.p = padding
        self.weight = np.random.randn(self.n_F, self.n_C, self.f, self.f) * np.sqrt(1. / (self.f))
        self.bias = np.random.randn(self.n_F) * np.sqrt(1. / self.n_F)
        
        self.weight_grad = np.zeros((self.n_F, self.n_C, self.f, self.f))
        self.bias_grad = np.zeros((self.n_F))


        self.cache = None

    def forward(self, X):
      
        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = self.n_F
        n_H = int((n_H_prev + 2 * self.p - self.f)/ self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f)/ self.s) + 1
        
        X_col = im2col(X, self.f, self.f, self.s, self.p)
        w_col = self.weight.reshape((self.n_F, -1))
        b_col = self.bias.reshape(-1, 1)
        
        out = w_col @ X_col + b_col
        
        out = np.array(np.hsplit(out, m)).reshape((m, n_C, n_H, n_W))
        self.cache = X, X_col, w_col
        return out

    def backward(self, dout):
        
        X, X_col, w_col = self.cache
        m, _, _, _ = X.shape
        self.bias_grad = np.sum(dout, axis=(0,2,3))
        dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = np.array(np.vsplit(dout, m))
        dout = np.concatenate(dout, axis=-1)
        dX_col = w_col.T @ dout
        dw_col = dout @ X_col.T
        dX = col2im(dX_col, X.shape, self.f, self.f, self.s, self.p)
        self.weight_grad = dw_col.reshape((dw_col.shape[0], self.n_C, self.f, self.f))
                
        return dX


class AvgPool():
    def __init__(self, filter_size, stride):
        self.f = filter_size
        self.s = stride
        self.cache = None

    def forward(self, X):
        
        m, n_C_prev, n_H_prev, n_W_prev = X.shape
        
        n_C = n_C_prev
        n_H = int((n_H_prev - self.f)/ self.s) + 1
        n_W = int((n_W_prev - self.f)/ self.s) + 1

        A_pool = np.zeros((m, n_C, n_H, n_W))
    
        for i in range(m):
            
            for c in range(n_C):

                for h in range(n_H):
                    h_start = h * self.s
                    h_end = h_start + self.f
                    
                    for w in range(n_W):
                        w_start = w * self.s
                        w_end = w_start + self.f
                        
                        A_pool[i, c, h, w] = np.mean(X[i, c, h_start:h_end, w_start:w_end])
        
        self.cache = X

        return A_pool

    def backward(self, dout):
        
        X = self.cache
        m, n_C, n_H, n_W = dout.shape
        dX = np.zeros(X.shape)        

        for i in range(m):
            
            for c in range(n_C):

                for h in range(n_H):
                    h_start = h * self.s
                    h_end = h_start + self.f

                    for w in range(n_W):
                        w_start = w * self.s
                        w_end = w_start + self.f

                        average = dout[i, c, h, w] / (self.f * self.f)
                        filter_average = np.full((self.f, self.f), average)
                        dX[i, c, h_start:h_end, w_start:w_end] += filter_average

        return dX