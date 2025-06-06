import torch
from torch import nn
from data_load import load_fashion_mnist_data
from train import train
from class_function_reused import try_gpu

#因为这里用的是fashion-mnist而不是ImageNet，因此初始通道数为1，并且最后一个全连接层的输出为10（本应是3和1000）
#输入图片大小是224*224
net=nn.Sequential(nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),nn.ReLU(),
                  nn.MaxPool2d(kernel_size=3,stride=2),
                  nn.Conv2d(96,256,kernel_size=5,padding=2),nn.ReLU(),
                  nn.MaxPool2d(kernel_size=3,stride=2),
                  nn.Conv2d(256,384,kernel_size=3,padding=1),nn.ReLU(),
                  nn.Conv2d(384,384,kernel_size=3,padding=1),nn.ReLU(),
                  nn.Conv2d(384,256,kernel_size=3,padding=1),nn.ReLU(),
                  nn.MaxPool2d(kernel_size=3,stride=2),
                  nn.Flatten(),
                  nn.Linear(6400,4096),nn.ReLU(),nn.Dropout(p=0.5),
                  nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(p=0.5),
                  nn.Linear(4096,10))

batch_size=128
train_iter,test_iter=load_fashion_mnist_data(batch_size,resize=224)
learning_rate=0.01
num_epochs=10

train(net,train_iter,test_iter,num_epochs,learning_rate,try_gpu())