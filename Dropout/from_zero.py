import torch
from torch import nn
from data_load import load_fashion_mnist_data
from class_function_reused import train
from class_function_reused import predict

#define dropout
def dropout_layer(X,dropout_param):
    assert 0<=dropout_param<=1
    if dropout_param==1:
        return torch.zeros_like(X)
    if dropout_param==0:
        return X
    mask=(torch.rand(X.shape)>dropout_param).float()
    return mask*X/(1.0-dropout_param)

#define model
class Net(nn.Module):
    def __init__(self,num_inputs,num_outputs,num_hiddens1,num_hiddens2,is_training=True):
        super().__init__()
        self.num_inputs=num_inputs
        self.is_training=is_training
        self.relu=nn.ReLU()
        self.lin1=nn.Linear(num_inputs,num_hiddens1)
        self.lin2=nn.Linear(num_hiddens1,num_hiddens2)
        self.lin3=nn.Linear(num_hiddens2,num_outputs)
    
    def forward(self,X):
        H1=self.relu(self.lin1(X.reshape((-1,self.num_inputs))))
        if self.is_training==True:
            H1=dropout_layer(H1,dropout_param1)
        H2=self.relu(self.lin2(H1))
        if self.is_training==True:
            H2=dropout_layer(H2,dropout_param2)
        outputs=self.lin3(H2)
        return outputs

num_inputs=784
num_outputs=10
num_hiddens1=256
num_hiddens2=256

dropout_param1=0.2
dropout_param2=0.5

num_epochs=10
learning_rate=0.5
batch_size=256

net=Net(num_inputs,num_outputs,num_hiddens1,num_hiddens2)
loss=nn.CrossEntropyLoss(reduction="none")
updater=torch.optim.SGD(net.parameters(),lr=learning_rate)
train_iter,test_iter=load_fashion_mnist_data(batch_size)

train(net,train_iter,test_iter,loss,updater,num_epochs)
predict(net,test_iter)
