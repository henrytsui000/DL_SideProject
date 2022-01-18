from .layer import *

class Network(object):
    def __init__(self):

        ## by yourself .Finish your own NN framework
        ## Just an example.You can alter sample code anywhere. 
        ## Just an example.You can alter sample code anywhere.

        self.conv1 = Conv2d(fil_num = 7, fil_size = 5, channel_num = 1) 
        self.relu1 = Leaky_ReLU()
        self.pool1 = AvgPool(fil_size = 2, stride = 2) 
        self.conv2 = Conv2d(fil_num = 15, fil_size = 7, channel_num = 7) 
        self.relu2 = Leaky_ReLU()
        self.pool2 = AvgPool(fil_size = 2, stride = 2)
        self.pool2_shape = None
        self.fc1 = FullyConnected(2160, 1024)
        self.fc2 = FullyConnected(1024, 256)
        self.fc3 = FullyConnected(256, 64)
        self.fc4 = FullyConnected(64, 13)
        
        self.act1 = Leaky_ReLU()
        self.act2 = Leaky_ReLU()
        self.act3 = Leaky_ReLU()
        
        self.loss = SoftmaxWithloss()
        self.target = np.empty([])
        self.SGD = np.empty([])
        
        self.optim1 = Adam()
        self.optim2 = Adam()
        self.optim3 = Adam()
        self.optim4 = Adam()
        self.optim5 = Adam()
        self.optim6 = Adam()
        self.optim7 = Adam()


    def forward(self, input, target):
        self.target = target

        input = input.reshape(input.shape[0], 1,int(np.sqrt(input.shape[1])), int(np.sqrt(input.shape[1])))
        conv1 = self.conv1.forward(input) 
        act1 = self.relu1.forward(conv1)
        pool1 = self.pool1.forward(act1)
        conv2 = self.conv2.forward(pool1)
        act2 = self.relu2.forward(conv2)
        pool2 = self.pool2.forward(act2)
        self.pool2_shape = pool2.shape #Need it in backpropagation.
        pool2_flatten = pool2.reshape(self.pool2_shape[0], -1) #(500*1024)


        h1 = self.fc1.forward(pool2_flatten)
        a1 = self.act1.forward(h1)
        h2 = self.fc2.forward(a1)
        a2 = self.act2.forward(h2)
        h3 = self.fc3.forward(a2)
        a3 = self.act3.forward(h3)
        
        h4 = self.fc4.forward(a3)
        
        pred, loss = self.loss.forward(h4, target)

        return pred, loss

    def backward(self):
        ## by yourself .Finish your own NN framework
        loss_grad = self.loss.backward(self.target)
        h4_grad = self.fc4.backward(loss_grad)
        a3_grad = self.act3.backward(h4_grad)
        h3_grad = self.fc3.backward(a3_grad)
        a2_grad = self.act2.backward(h3_grad)
        h2_grad = self.fc2.backward(a2_grad)
        a1_grad = self.act1.backward(h2_grad)
        h1_grad = self.fc1.backward(a1_grad)

        h1_grad = h1_grad.reshape(self.pool2_shape)
        conv_grad = self.pool2.backward(h1_grad) 
        conv_grad = self.relu2.backward(conv_grad) 
        conv_grad= self.conv2.backward(conv_grad)
        conv_grad = self.pool1.backward(conv_grad) 
        conv_grad = self.relu1.backward(conv_grad)
        conv_grad = self.conv1.backward(conv_grad)


    def update(self, lr):
        ## by yourself .Finish your own NN framework
        '''
        self.fc1.weight -= lr * self.fc1.weight_grad
        self.fc1.bias -= lr * self.fc1.bias_grad
        self.fc2.weight -= lr * self.fc2.weight_grad
        self.fc2.bias -= lr * self.fc2.bias_grad
        self.fc3.weight -= lr * self.fc3.weight_grad
        self.fc3.bias -= lr * self.fc3.bias_grad
        self.fc4.weight -= lr * self.fc4.weight_grad
        self.fc4.bias -= lr * self.fc4.bias_grad
        self.fc5.weight -= lr * self.fc5.weight_grad
        self.fc5.bias -= lr * self.fc5.bias_grad
        self.fc6.weight -= lr * self.fc6.weight_grad
        self.fc6.bias -= lr * self.fc6.bias_grad
        self.fc7.weight -= lr * self.fc7.weight_grad
        self.fc7.bias -= lr * self.fc7.bias_grad
        '''
        self.optim1.update(self.fc1,lr)
        self.optim2.update(self.fc2,lr)
        self.optim3.update(self.fc3,lr)
        self.optim4.update(self.fc4,lr)
        self.optim5.update(self.conv1,lr)
        self.optim6.update(self.conv2,lr)
        
        
        