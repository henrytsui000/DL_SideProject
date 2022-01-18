import numpy as np


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
        

class FullyConnected(_Layer):
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((1, out_features))
        self.input = np.empty([])
        self.weight_grad = np.empty([])
        self.bias_grad = np.empty([])

    def forward(self, input):
        output = np.dot(input, self.weight) + self.bias
        self.input = input
        return output

    def backward(self, output_grad):
        input_grad = np.dot(output_grad, self.weight.T)
        self.weight_grad = (np.dot(self.input.T, output_grad))
        self.bias_grad = output_grad.mean(axis=0)
        return input_grad


class SoftmaxWithloss(_Layer):
    def __init__(self):
        self.predict = np.empty([])
        
    def forward(self, input, target):
        '''Softmax'''     
        for i in range(len(input)):
            dummy = np.exp(input[i]-np.max(input[i]))
            input[i] = dummy / np.sum(dummy)
        self.predict = input

        '''Average loss'''
        col = target.shape[0]
        sum_reg = -np.log(self.predict)*target
        your_loss = np.sum(sum_reg) / col
        return self.predict, your_loss

    def backward(self, target):
        input_grad = self.predict - target 
        return input_grad

class Leaky_ReLU(_Layer):
    def __init__(self):
        self.input = np.empty([])

    def forward(self, input): 
        self.input = input
        output = input.copy()
        output[self.input<0] *= 0.01
        return output

    def backward(self, output_grad):
        output_grad[self.input<0] *= 0.01
        return output_grad

class Sigmoid(_Layer):
    def __init__(self):
        self.input = np.empty([])

    def forward(self, input):
        self.input = input
        return 1 / (1 + np.exp(input*-1))
    
    def backward(self, output_grad):
        reg = 1 / (1 + np.exp(self.input*-1))
        return output_grad*reg*(1 - reg)
    
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
        
        fc_obj.weight -= lr*(m_wgrad_corr/(np.sqrt(v_wgrad_corr)+self.eps))
        fc_obj.bias -= lr*(m_bgrad_corr/(np.sqrt(v_bgrad_corr)+self.eps))
        self.t += 1

def get_indices(X_shape, input_H, input_W, stride, pad):
    m, column_num, height, weight = X_shape
    output_h = int((height + 2 * pad - input_H) / stride) + 1
    output_w = int((weight + 2 * pad - input_W) / stride) + 1
    level1 = np.repeat(np.arange(input_H), input_W)
    level1 = np.tile(level1, column_num)
    everyLevels = stride * np.repeat(np.arange(output_h), output_w)
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    tile1 = np.tile(np.arange(input_W), input_H)
    tile1 = np.tile(tile1, column_num)
    everySlides = stride * np.tile(np.arange(output_w), output_h)
    j = tile1.reshape(-1, 1) + everySlides.reshape(1, -1)
    d = np.repeat(np.arange(column_num), input_H * input_W).reshape(-1, 1)

    return i, j, d

def im2col(X, input_H, input_W, stride, pad):
    
    X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, input_H, input_W, stride, pad)
    ret = X_padded[:, d, i, j]
    ret = np.concatenate(ret, axis=-1)
    return ret


def col2im(dX_col, X_shape, input_H, input_W, stride, pad):
    
    
    N, D, H, W = X_shape
    new_H, new_W = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((N, D, new_H, new_W))
    
    i, j, d = get_indices(X_shape, input_H, input_W, stride, pad)
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))
    np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[pad:-pad, pad:-pad, :, :]
    
class Conv2d(_Layer):
    
    def __init__(self, fil_num, fil_size, channel_num, stride=1, padding=0):
        self.fil_num = fil_num
        self.f = fil_size
        self.n_C = channel_num
        self.s = stride
        self.p = padding
        self.weight = np.random.randn(self.fil_num, self.n_C, self.f, self.f) * np.sqrt(1. / (self.f))
        self.bias = np.random.randn(self.fil_num) * np.sqrt(1. / self.fil_num)
        
        self.weight_grad = np.zeros((self.fil_num, self.n_C, self.f, self.f))
        self.bias_grad = np.zeros((self.fil_num))


        self.reg = None

    def forward(self, X):
      
        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        column_num = self.fil_num
        height = int((2 * self.p + n_H_prev - self.f)/ self.s) + 1
        width = int(( 2 * self.p + n_W_prev - self.f)/ self.s) + 1
        
        X_col = im2col(X, self.f, self.f, self.s, self.p)
        w_col = self.weight.reshape((self.fil_num, -1))
        b_col = self.bias.reshape(-1, 1)
        
        out = w_col @ X_col + b_col
        
        out = np.array(np.hsplit(out, m)).reshape((m, column_num, height, width))
        self.reg = X, X_col, w_col
        return out

    def backward(self, output_grad):
        
        X, X_col, w_col = self.reg
        first_dim, _, _, _ = X.shape
        self.bias_grad = np.sum(output_grad, axis=(0,2,3))
        output_grad = output_grad.reshape(output_grad.shape[0] * output_grad.shape[1], output_grad.shape[2] * output_grad.shape[3])
        output_grad = np.array(np.vsplit(output_grad, first_dim))
        output_grad = np.concatenate(output_grad, axis=-1)
        dX_col = w_col.T @ output_grad
        dw_col = output_grad @ X_col.T
        input_grad = col2im(dX_col, X.shape, self.f, self.f, self.s, self.p)
        self.weight_grad = dw_col.reshape((dw_col.shape[0], self.n_C, self.f, self.f))

        return input_grad


class AvgPool():
    
    def __init__(self, fil_size, stride):
        self.f = fil_size
        self.s = stride
        self.reg = None

    def forward(self, X):
        
        m, X_C, X_H, X_W = X.shape
        column_num = X_C
        height = int((X_H - self.f)/ self.s) + 1
        width = int((X_W - self.f)/ self.s) + 1
        A_pool = np.zeros((m, column_num, height, width))

        for i in range(m):
            for j in range(column_num):
                for h in range(height):
                    h_start = h * self.s
                    h_end = h_start + self.f
                    for w in range(width):
                        w_start = w * self.s
                        w_end = w_start + self.f
                        A_pool[i, j, h, w] = np.mean(X[i, j, h_start:h_end, w_start:w_end])
        
        self.reg = X
        return A_pool

    def backward(self, output_grad):
        
        X = self.reg
        m, n_C, n_H, n_W = output_grad.shape
        input_grad = np.zeros(X.shape)        

        for i in range(m):
            for c in range(n_C):
                for h in range(n_H):
                    h_start = h * self.s
                    h_end = h_start + self.f

                    for w in range(n_W):
                        w_start = w * self.s
                        w_end = w_start + self.f

                        average = output_grad[i, c, h, w] / (self.f * self.f)
                        filter_average = np.full((self.f, self.f), average)
                        input_grad[i, c, h_start:h_end, w_start:w_end] += filter_average

        return input_grad

