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

#read dataset
def load_data(features,labels,batch_size,is_train=True):
    dataset=data.TensorDataset(features,labels)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

def evaluate_loss(net,data_iter,loss):
    metric=Accumulator(2)
    for x,y in data_iter:
        temp_outputs=net(x)
        y=y.reshape(temp_outputs.shape)
        temp_loss=loss(temp_outputs,y)
        metric.add(temp_loss.sum(),y.numel())
    return metric[0]/metric[1]