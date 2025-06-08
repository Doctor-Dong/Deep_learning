import torch
from torch import nn
from torch.nn import functional
from data_load import load_fashion_mnist_data
from train import train
from class_function_reused import try_gpu

class Residual(nn.Module):
    def __init__(self,input_channels,num_channels,use_conv=False,stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(input_channels,num_channels,kernel_size=3,stride=stride,padding=1)
        self.conv2=nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)
        if use_conv:
            self.conv3=nn.Conv2d(input_channels,num_channels,kernel_size=1,stride=stride)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)

    def forward(self,X):
        Y=functional.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        Y+=X
        return functional.relu(Y)
    
def resnet_block(input_channels,num_channels,num_residuals,first_block=False):
    block=[]
    for i in range(0,num_residuals):
        if i==0 and not first_block:
            block.append(Residual(input_channels,num_channels,use_conv=True,stride=2))
        else:
            block.append(Residual(num_channels,num_channels))
    return block

#define blocks
block1=nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
                     nn.BatchNorm2d(64),nn.ReLU(),
                     nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
block2=nn.Sequential(*resnet_block(64,64,2,first_block=True))
block3=nn.Sequential(*resnet_block(64,128,2))
block4=nn.Sequential(*resnet_block(128,256,2))
block5=nn.Sequential(*resnet_block(256,512,2))

#define the net
net=nn.Sequential(block1,block2,block3,block4,block5,
                  nn.AdaptiveAvgPool2d((1,1)),
                  nn.Flatten(),
                  nn.Linear(512,10))

learning_rate=0.05
num_epochs=10
batch_size=256
train_iter,test_iter=load_fashion_mnist_data(batch_size,resize=96)

train(net,train_iter,test_iter,num_epochs,learning_rate,try_gpu())