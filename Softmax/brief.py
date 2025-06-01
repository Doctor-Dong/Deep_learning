import torch 
from torch import nn
from data_load import load_fashion_mnist_data
from class_function_reused import Accumulator
from from_zero import train

def init_weights(module):
    if type(module)==nn.Linear:
        nn.init.normal_(module.weight,mean=0.0,std=0.01)

batch_size=256
train_iter,test_iter=load_fashion_mnist_data(batch_size)

#define model
net=nn.Sequential(nn.Flatten(),nn.Linear(784,10))

#init weights
net.apply(init_weights)

loss=nn.CrossEntropyLoss(reduction="none")
trainer=torch.optim.SGD(net.parameters(),lr=0.1)
num_epochs=10

train(net,train_iter,test_iter,loss,trainer,num_epochs)