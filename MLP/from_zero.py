import torch
from torch import nn
from data_load import load_fashion_mnist_data
from class_function_reused import train
from class_function_reused import predict

#define relu
def relu(X):
    temp=torch.zeros_like(X)
    return torch.max(X,temp)

#define net
def net(X):
    X=X.reshape((-1,num_inputs))
    temp=relu(torch.matmul(X,weights1)+bias1)
    return torch.matmul(temp,weights2)+bias2

#define loss
loss=nn.CrossEntropyLoss(reduction="none")

batch_size=256
train_iter,test_iter=load_fashion_mnist_data(batch_size)
num_inputs,num_outputs,num_hiddens=784,10,256

#define params
weights1=nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True)*0.01)
bias1=nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))
weights2=nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True)*0.01)
bias2=nn.Parameter(torch.zeros(num_outputs,requires_grad=True))
params=[weights1,bias1,weights2,bias2]

num_epochs=10
learning_rate=0.1
updater=torch.optim.SGD(params,lr=learning_rate)

train(net,train_iter,test_iter,loss,updater,num_epochs)
predict(net,test_iter)
