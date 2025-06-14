import time
import numpy
import torch

class Timer:
    def __init__(self):
        self.times=[]

    def start(self):
        self.start_time=time.time()

    def stop(self): #记录并返回这段时长
        self.times.append(time.time()-self.start_time)
        return self.times[-1]
    
    def average(self):
        return sum(self.times)/len(self.times)
    
    def total_time(self):
        return sum(self.times)
    
    def cumsum_time(self):
        return numpy.array(self.times).cumsum().tolist()
    
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