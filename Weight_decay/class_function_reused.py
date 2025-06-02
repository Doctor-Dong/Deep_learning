import torch
from torch.utils import data

class Accumulator:
    def __init__(self,n):
        self.data=[0.0]*n

    def __getitem__(self,index):
        return self.data[index]
    
    def add(self,*params):
        self.data=[a+float(b) for a,b in zip(self.data,params) ]

    def reset(self):
        self.data=[0.0]*len(self.data)

def synthetic_data(weight,bias,num_examples):
    x=torch.normal(0,1,(num_examples,len(weight)))
    y=torch.matmul(x,weight)+bias
    y+=torch.normal(0,0.01,y.shape) #随机噪声
    return x,y.reshape((-1,1))

def load_data(features,labels,batch_size,is_train=True):
    dataset=data.TensorDataset(features,labels)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

def linreg(x,weights,bias):
    return torch.matmul(x,weights)+bias

def squared_loss(y_hat,y):
    return ((y_hat-y.reshape(y_hat.shape))**2)/2

def sgd(params,learning_rate,batch_size):
    with torch.no_grad():
        for param in params:
            param-=learning_rate*param.grad/batch_size
            param.grad.zero_()