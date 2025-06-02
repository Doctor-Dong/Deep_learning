import torch
from torch import nn
from data_load import load_fashion_mnist_data
from class_function_reused import train
from class_function_reused import predict

#define net
net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10))

def init_weights(module):
    if type(module)==nn.Linear:
        nn.init.normal_(module.weight,std=0.01)

#init weights
net.apply(init_weights)

batch_size=256
learning_rate=0.1
num_epochs=10

loss=nn.CrossEntropyLoss(reduction="none")
updater=torch.optim.SGD(net.parameters(),lr=learning_rate)
train_iter,test_iter=load_fashion_mnist_data(batch_size)

train(net,train_iter,test_iter,loss,updater,num_epochs)
predict(net,test_iter)