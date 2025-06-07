import torch
from torch import nn
from data_load import load_fashion_mnist_data
from class_function_reused import try_gpu
from train import train

net=nn.Sequential(nn.Conv2d(1,6,kernel_size=5),nn.BatchNorm2d(6),nn.Sigmoid(),
                  nn.AvgPool2d(kernel_size=2,stride=2),
                  nn.Conv2d(6,16,kernel_size=5),nn.BatchNorm2d(16),nn.Sigmoid(),
                  nn.AvgPool2d(kernel_size=2,stride=2),
                  nn.Flatten(),
                  nn.Linear(16*4*4,120),nn.BatchNorm1d(120),nn.Sigmoid(),
                  nn.Linear(120,84),nn.BatchNorm1d(84),nn.Sigmoid(),
                  nn.Linear(84,10))

learning_rate=1.0
num_epochs=10
batch_size=256
train_iter,test_iter=load_fashion_mnist_data(batch_size)

train(net,train_iter,test_iter,num_epochs,learning_rate,try_gpu())