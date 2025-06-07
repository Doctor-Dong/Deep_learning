import torch
from torch import nn
from data_load import load_fashion_mnist_data
from class_function_reused import try_gpu
from train import train

class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4):
        super().__init__()
        self.p1_1=nn.Conv2d(in_channels,c1,kernel_size=1)
        self.p2_1=nn.Conv2d(in_channels,c2[0],kernel_size=1)
        self.p2_2=nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)
        self.p3_1=nn.Conv2d(in_channels,c3[0],kernel_size=1)
        self.p3_2=nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)
        self.p4_1=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.p4_2=nn.Conv2d(in_channels,c4,kernel_size=1)
        self.relu=nn.ReLU()

    def forward(self,x):
        p1=self.relu(self.p1_1(x))
        p2=self.relu(self.p2_2(self.relu(self.p2_1(x))))
        p3=self.relu(self.p3_2(self.relu(self.p3_1(x))))
        p4=self.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1,p2,p3,p4),dim=1)

#define the blocks
block1=nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
block2=nn.Sequential(nn.Conv2d(64,64,kernel_size=1),
                     nn.ReLU(),
                     nn.Conv2d(64,192,kernel_size=3,padding=1),
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
block3=nn.Sequential(Inception(192,64,(96,128),(16,32),32),
                     Inception(256,128,(128,192),(32,96),64),
                     nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
block4=nn.Sequential(Inception(480,192,(96,208),(16,48),64),
                     Inception(512,160,(112,224),(24,64),64),
                     Inception(512,128,(128,256),(24,64),64),
                     Inception(512,112,(144,288),(32,64),64),
                     Inception(528,256,(160,320),(32,128),128),
                     nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
block5=nn.Sequential(Inception(832,256,(160,320),(32,128),128),
                     Inception(832,384,(192,384),(48,128),128),
                     nn.AdaptiveAvgPool2d((1,1)),
                     nn.Flatten())

#define the net
net=nn.Sequential(block1,block2,block3,block4,block5,nn.Linear(1024,10))

learning_rate=0.1
num_epochs=10
batch_size=128

train_iter,test_iter=load_fashion_mnist_data(batch_size,resize=96)

train(net,train_iter,test_iter,num_epochs,learning_rate,try_gpu())