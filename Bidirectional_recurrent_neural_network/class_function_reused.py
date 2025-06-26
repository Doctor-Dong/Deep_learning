import torch
from torch import nn
from torch.nn import functional
import time
import numpy
import math

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

class RNN_Model:
    def __init__(self,vocab_size,num_hiddens,device,get_params,init_state,forward_function):
        self.vocab_size=vocab_size
        self.num_hiddens=num_hiddens
        self.params=get_params(vocab_size,num_hiddens,device)
        self.init_state=init_state
        self.forward_function=forward_function

    def __call__(self,X,state):
        X=functional.one_hot(X.T,self.vocab_size).type(torch.float32)
        return self.forward_function(X,state,self.params)
    
    def begin_state(self,batch_size,device):
        return self.init_state(batch_size,self.num_hiddens,device)
    
class RNNModel(nn.Module):  #for brief
    def __init__(self,rnn_layer,vocab_size):
        super().__init__()
        self.rnn=rnn_layer
        self.vocab_size=vocab_size
        self.num_hiddens=self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions=1
            self.linear=nn.Linear(self.num_hiddens,self.vocab_size)
        else:
            self.num_directions=2
            self.linear=nn.Linear(2*self.num_hiddens,self.vocab_size)

    def forward(self,inputs,state):
        X=functional.one_hot(inputs.T.long(),self.vocab_size)
        X=X.to(torch.float32)
        Y,state=self.rnn(X,state)
        output=self.linear(Y.reshape((-1,Y.shape[-1])))
        return output,state
    
    def begin_state(self,device,batch_size=1):
        if not isinstance(self.rnn,nn.LSTM):
            return torch.zeros((self.num_directions*self.rnn.num_layers,batch_size,self.num_hiddens),device=device)
        else:
            return (torch.zeros((self.num_directions*self.rnn.num_layers,batch_size,self.num_hiddens),device=device),torch.zeros((self.num_directions*self.rnn.num_layers,batch_size,self.num_hiddens),device=device))

    
def grad_clipping(net,theta):
    if isinstance(net,nn.Module):
        params=[p for p in net.parameters() if p.requires_grad]
    else:
        params=net.params
    norm=torch.sqrt(sum(torch.sum(param.grad**2) for param in params))
    if norm>theta:
        for param in params:
            param.grad[:]*=theta/norm

def predict(prefix,num_preds,net,vocab,device):
    state=net.begin_state(batch_size=1,device=device)
    outputs=[vocab[prefix[0]]]
    get_input=lambda:torch.tensor([outputs[-1]],device=device).reshape((1,1))
    for y in prefix[1:]:
        _,state=net(get_input(),state)
        outputs.append(vocab[y])
    for _ in range(0,num_preds):
        y,state=net(get_input(),state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return "".join([vocab.index_to_token[i] for i in outputs])
    
def train_epoch(net,train_iter,loss,updater,device,use_random_iter):
    state=None
    timer=Timer()
    timer.start()
    metric=Accumulator(2)
    for X,Y in train_iter:
        if state is None or use_random_iter:
            state=net.begin_state(batch_size=X.shape[0],device=device)
        else:
            if isinstance(net,nn.Module) and not isinstance(state,tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y=Y.T.reshape(-1)
        X=X.to(device)
        y=y.to(device)
        y_hat,state=net(X,state)
        temp_loss=loss(y_hat,y.long()).mean()
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            temp_loss.backward()
            grad_clipping(net,theta=1)
            updater.step()
        else:
            temp_loss.backward()
            grad_clipping(net,theta=1)
            updater(batch_size=1)
        metric.add(temp_loss*y.numel(),y.numel())
    return math.exp(metric[0]/metric[1]),metric[1]/timer.stop()

def train(net,train_iter,vocab,learning_rate,num_epochs,device,use_random_iter=False):
    loss=nn.CrossEntropyLoss()
    if isinstance(net,nn.Module):
        updater=torch.optim.SGD(net.parameters(),lr=learning_rate)
    else:
        updater=lambda batch_size:sgd(net.params,learning_rate=learning_rate,batch_size=batch_size)
    temp_predict=lambda prefix:predict(prefix,num_preds=50,net=net,vocab=vocab,device=device)
    for epoch in range(0,num_epochs):
        perplexity,speed=train_epoch(net,train_iter,loss,updater,device,use_random_iter)
        if (epoch+1)%10==0:
            print(temp_predict("time traveller "))
    print(f"Perplexity={perplexity:.1f},speed={speed:.1f}/second,device={str(device)}")
    print(temp_predict("time traveller "))
    print(temp_predict("traveller"))