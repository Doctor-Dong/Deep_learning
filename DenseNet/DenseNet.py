import torch
from torch import nn
from data_load import load_fashion_mnist_data
from class_function_reused import try_gpu
from train import train

def conv_block(input_channels,num_channels):
    return nn.Sequential(nn.BatchNorm2d(input_channels),
                         nn.ReLU(),
                         nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1))

class DenseBlock(nn.Module):
    def __init__(self,num_convs,input_channels,num_channels):
        super().__init__()
        block=[]
        for i in range(0,num_convs):
            block.append(conv_block(num_channels*i+input_channels,num_channels))
        self.net=nn.Sequential(*block)
    
    def forward(self,X):
        for blk in self.net:
            Y=blk(X)
            X=torch.cat((X,Y),dim=1)
        return X
    
def transition_block(input_channels,num_channels):
    return nn.Sequential(nn.BatchNorm2d(input_channels),
                         nn.ReLU(),
                         nn.Conv2d(input_channels,num_channels,kernel_size=1),
                         nn.AvgPool2d(kernel_size=2,stride=2))

#define blocks
block1=nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
                     nn.BatchNorm2d(64),nn.ReLU(),
                     nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

num_channels=64
growth_rate=32  #充当先前函数的num_channels参数
num_convs_in_denseblocks=[4,4,4,4]
blocks=[]
for index,num_convs in enumerate(num_convs_in_denseblocks):
    blocks.append(DenseBlock(num_convs,num_channels,growth_rate))
    num_channels+=growth_rate*num_convs
    if index!=len(num_convs_in_denseblocks)-1:
        blocks.append(transition_block(num_channels,num_channels//2))
        num_channels=num_channels//2

#define the net
net=nn.Sequential(block1,*blocks,
                  nn.BatchNorm2d(num_channels),nn.ReLU(),
                  nn.AdaptiveAvgPool2d((1,1)),
                  nn.Flatten(),
                  nn.Linear(num_channels,10))

learning_rate=0.1
num_epochs=10
batch_size=256
train_iter,test_iter=load_fashion_mnist_data(batch_size,resize=96)

train(net,train_iter,test_iter,num_epochs,learning_rate,try_gpu())