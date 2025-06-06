import torch
from torch import nn
from data_load import load_fashion_mnist_data
from train import train
from class_function_reused import try_gpu

def vgg_block(num_convs,in_channels,out_channels):
    layers=[]
    for _ in range(0,num_convs):
        layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        layers.append(nn.ReLU())
        in_channels=out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def vgg(conv_nums_outchannels):
    conv_blocks=[]
    in_channels=1
    for (num_convs,out_channels) in conv_nums_outchannels:
        conv_blocks.append(vgg_block(num_convs,in_channels,out_channels))
        in_channels=out_channels
    return nn.Sequential(*conv_blocks,nn.Flatten(),
                         nn.Linear(out_channels*7*7,4096),nn.ReLU(),nn.Dropout(p=0.5),
                         nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(p=0.5),
                         nn.Linear(4096,10))

conv_nums_outchannels=((1,64),(1,128),(2,256),(2,512),(2,512))

#Vgg11=vgg(conv_nums_outchannels)
#too hard to train,reduce the number of channels

ratio=4
smaller=[(block[0],block[1]//ratio) for block in conv_nums_outchannels]
net=vgg(smaller)

learning_rate=0.05
num_epochs=10
batch_size=128
train_iter,test_iter=load_fashion_mnist_data(batch_size,resize=224)

train(net,train_iter,test_iter,num_epochs,learning_rate,try_gpu())


