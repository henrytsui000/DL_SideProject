import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class HandWritten(Dataset):
    def __init__(self, adata, alabel, transform=None):
        self.transform = transform
        mydata = np.loadtxt(adata,delimiter=',',dtype="float32")
        Means = np.mean(mydata, axis=1)
        Std = np.std(mydata, ddof=1,axis=1)
        MEANS, STD= np.repeat(Means , 4096), np.repeat(Std, 4096)
        MEANS, STD =MEANS.reshape(11500, -1), STD.reshape(11500, -1)
        print(MEANS.shape, STD.shape)

        out = (mydata-MEANS)/STD
        self.data = (out/2)+0.5
        self.data = self.data.reshape(11500,1,64,64)
        self.label = np.loadtxt(alabel,delimiter=',',dtype="int")
        print('done')
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if self.transform:
            self.data[idx] = self.transform(self.data[idx]).reshape(1,64,64)
        sample = {"Image":torch.from_numpy(self.data[idx]), "Class": self.label[idx]}
        return sample

    def getnum(self):
        return self.data.shape[0]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 7, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(7, 15, 7)
        self.fc1 = nn.Linear(2160, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 13)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.act4 = nn.ReLU()
        self.act5 = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool1(self.act2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        x = self.act5(self.fc3(x))
        x = self.fc4(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCH = 20
Batch_size = 125
Learning_rate = 1e-3

Transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

dataset = HandWritten('./data/train_data.csv','./data/train_label.csv')
train_set, val_set = torch.utils.data.random_split(dataset, [10000, 1500])
trainloader = DataLoader(train_set, batch_size=Batch_size, shuffle=True, num_workers=2)
validloader = DataLoader(val_set, batch_size=Batch_size)

train_batch_num = 10000 / Batch_size
valid_batch_num = 1500 / Batch_size
net = Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=Learning_rate)


writer = SummaryWriter('runs/train_converge')
classes = ('0','1','2','3','4','5','6','7','8','9','10','1e2','1e3')
def images_to_probs(net, images):

    output = net(images)
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig
for epoch in range(EPOCH):
    running_loss = 0.0
    val_loss = 0.0
    val_image_num = 0
    val_hit = 0
    train_hit = 0
    train_image_num = 0
    a = 0
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data['Image'].to(device), data['Class'].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        train_image_num += inputs.shape[0]
        for i in range(len(outputs)):
            if(np.argmax(outputs[i].cpu().detach().numpy())==labels[i]):
                train_hit += 1
        if i % 125 == 124:    # every 1000 mini-batches...
            writer.add_scalar('training loss',
                            running_loss / 125,
                            epoch * len(trainloader) + i)
            writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(net, inputs, labels),
                            global_step=epoch * len(trainloader) + i)
            running_loss = 0.0
    for data in validloader:
        inputs, labels = data['Image'].to(device), data['Class'].to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        val_image_num += inputs.shape[0]
        for i in range(len(outputs)):
            if(np.argmax(outputs[i].cpu().detach().numpy())==labels[i]):
                val_hit += 1
    
    print('Epoch:%3d'%epoch, '|Train Loss:%8.4f'%(running_loss/train_batch_num), '|Train Acc:%3.4f'%(train_hit/(train_image_num)*100.0))
    print('Epoch:%3d'%epoch, '|Valid Loss:%8.4f'%(val_loss/valid_batch_num), '|Valid Acc:%3.4f'%(val_hit/(val_image_num)*100.0))
