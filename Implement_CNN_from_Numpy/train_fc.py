import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
import torchvision.transforms as transforms
import pandas as pd


'''
self.transform = transform
        self.data = np.loadtxt(adata,delimiter=',',dtype="float32")
        self.data = self.data.reshape(self.data.shape[0], 1,64, 64)
        self.label = np.loadtxt(alabel,delimiter=',',dtype="int")'''


class HandWritten(Dataset):
    def __init__(self, adata, alabel, transform=None):
        self.transform = transform
        '''mydata = np.loadtxt(adata,delimiter=',',dtype="float32")

        Means = np.mean(mydata, axis=1)
        Std = np.std(mydata, ddof=1,axis=1)

        MEANS, STD= np.repeat(Means , 4096), np.repeat(Std, 4096)

        MEANS, STD =MEANS.reshape(11500, -1), STD.reshape(11500, -1)

        print(MEANS.shape, STD.shape)

        out = (mydata-MEANS)/STD
        self.data = (out/2)+0.5'''
        self.data = np.loadtxt(adata,delimiter=',',dtype="float32")
        self.label = np.loadtxt(alabel,delimiter=',',dtype="int")
        print('done')
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if self.transform:
            self.data[idx] = self.transform(self.data[idx]).reshape(1,64,64)
        
        #dummy = np.eye(13)[int(self.label[idx])]

        sample = {"Image":torch.from_numpy(self.data[idx]), "Class": self.label[idx]}

        return sample

    def getnum(self):
        return self.data.shape[0]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 1000)
        self.fc3 = nn.Linear(1000, 765)
        self.fc4 = nn.Linear(765, 328)
        self.fc5 = nn.Linear(328, 256)
        self.fc6 = nn.Linear(256, 64)
        self.fc7 = nn.Linear(64, 13)
        

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        return x

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
EPOCH = 10
Batch_size = 250
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

'''
cnt = 0
for data in trainloader:
    inputs, labels = data['Image'].to(device), data['Class'].to(device)
    if(cnt == 0):
        print(data['Image'][0], data['Image'].shape)'''


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
        #print(net.fc3.weight.grad)
        optimizer.step()

        running_loss += loss.item()

        train_image_num += inputs.shape[0]
        for i in range(len(outputs)):
            if(np.argmax(outputs[i].cpu().detach().numpy())==labels[i]):
                train_hit += 1
        
        #print(train_hit, train_image_num)
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
    
