import torch
from torch import nn
from class_function_reused import train
from class_function_reused import predict
from data_load import load_fashion_mnist_data

def init_weights(module):
    if type(module)==nn.Linear:
        nn.init.normal_(module.weight,std=0.01)

dropout_param1=0.2
dropout_param2=0.5
learning_rate=0.5
batch_size=256
num_epochs=10

train_iter,test_iter=load_fashion_mnist_data(batch_size)

net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Dropout(dropout_param1),nn.Linear(256,256),nn.ReLU(),nn.Dropout(dropout_param2),nn.Linear(256,10))
net.apply(init_weights)
updater=torch.optim.SGD(net.parameters(),lr=learning_rate)
loss=nn.CrossEntropyLoss(reduction="none")

train(net,train_iter,test_iter,loss,updater,num_epochs)
predict(net,test_iter)