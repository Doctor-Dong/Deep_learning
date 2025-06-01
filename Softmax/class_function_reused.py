import torch

class Accumulator:
    def __init__(self,n):
        self.data=[0.0]*n

    def __getitem__(self,index):
        return self.data[index]
    
    def add(self,*params):
        self.data=[a+float(b) for a,b in zip(self.data,params) ]

    def reset(self):
        self.data=[0.0]*len(self.data)

def sgd(params,learning_rate,batch_size):
    with torch.no_grad():
        for param in params:
            param-=learning_rate*param.grad/batch_size
            param.grad.zero_()