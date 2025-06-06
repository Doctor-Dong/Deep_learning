import torch
from torch import nn
from train import train
from data_load import load_fashion_mnist_data
from class_function_reused import try_gpu

def NiN_block(in_channels,out_channels,kernel_size,stride,padding):
    return nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
                         nn.ReLU(),
                         nn.Conv2d(out_channels,out_channels,kernel_size=1),nn.ReLU(),
                         nn.Conv2d(out_channels,out_channels,kernel_size=1),nn.ReLU())

net=nn.Sequential(NiN_block(1,96,kernel_size=11,stride=4,padding=0),
                  nn.MaxPool2d(kernel_size=3,stride=2),
                  NiN_block(96,256,kernel_size=5,stride=1,padding=2),
                  nn.MaxPool2d(kernel_size=3,stride=2),
                  NiN_block(256,384,kernel_size=3,stride=1,padding=1),
                  nn.MaxPool2d(kernel_size=3,stride=2),
                  nn.Dropout(p=0.5),
                  NiN_block(384,10,kernel_size=3,stride=1,padding=1),
                  nn.AdaptiveAvgPool2d((1,1)),
                  nn.Flatten())

learning_rate=0.1
num_epochs=10
batch_size=128
train_iter,test_iter=load_fashion_mnist_data(batch_size,resize=224)
train(net,train_iter,test_iter,num_epochs,learning_rate,try_gpu())