
from .layer import *

class Network(object):
    def __init__(self):

        ## by yourself .Finish your own NN framework
        ## Just an example.You can alter sample code anywhere. 
         ## Just an example.You can alter sample code anywhere. 
        
        
        self.conv1 = Conv(nb_filters = 6, filter_size = 5, nb_channels = 1)
        self.relu1 = Leaky_ReLU()
        self.pool1 = AvgPool(filter_size = 2, stride = 2)
        self.conv2 = Conv(nb_filters = 16, filter_size = 5, nb_channels = 6)
        self.relu2 = Leaky_ReLU()
        self.pool2 = AvgPool(filter_size = 2, stride = 2)
        self.pool2_shape = None
        
        self.fc1 = FullyConnected(2704, 1024)
        self.fc2 = FullyConnected(1024, 256)
        self.fc3 = FullyConnected(256, 64)
        self.fc4 = FullyConnected(64, 13)
        
        self.act1 = Leaky_ReLU()
        self.act2 = Leaky_ReLU()
        self.act3 = Leaky_ReLU()
        self.act4 = Leaky_ReLU()
        
        self.optim1 = Adam()
        self.optim2 = Adam()
        self.optim3 = Adam()
        self.optim4 = Adam()
        self.optim5 = Adam()
        self.optim6 = Adam()
        
        self.loss = SoftmaxWithloss()
        self.target = np.empty([])



    def forward(self, input, target):
        self.target = target
        
        input = input.reshape(input.shape[0], 1,int(np.sqrt(input.shape[1])), int(np.sqrt(input.shape[1])))
        conv1 = self.conv1.forward(input) 
        act1 = self.relu1.forward(conv1)
        pool1 = self.pool1.forward(act1) 

        conv2 = self.conv2.forward(pool1) 
        act2 = self.relu2.forward(conv2)
        pool2 = self.pool2.forward(act2)
        self.pool2_shape = pool2.shape 

        pool2_flatten = pool2.reshape(self.pool2_shape[0], -1)
        
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
        grad = self.loss.backward(self.target)
        
        grad = self.fc4.backward(grad)
        grad = self.act3.backward(grad)

        grad = self.fc3.backward(grad)
        grad = self.act2.backward(grad)

        grad = self.fc2.backward(grad)
        grad = self.act1.backward(grad)

        grad = self.fc1.backward(grad)
       
        grad = grad.reshape(self.pool2_shape)

        deltaL = self.pool2.backward(grad) 
        deltaL = self.relu2.backward(deltaL) 
        deltaL = self.conv2.backward(deltaL)

        deltaL = self.pool1.backward(deltaL) 
        deltaL = self.relu1.backward(deltaL)
        deltaL = self.conv1.backward(deltaL)
        

    def update(self, lr):
        ## by yourself .Finish your own NN framework
        
        self.optim1.update(self.fc1,lr)
        self.optim2.update(self.fc2,lr)
        self.optim3.update(self.fc3,lr)
        self.optim4.update(self.fc4,lr)
        
        self.optim5.update(self.conv1, lr)
        self.optim6.update(self.conv2, lr)
        