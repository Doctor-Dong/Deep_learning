import torch
from torch import nn
from dataset import load_data_time_machine
from torch.nn import functional
from GPU import try_gpu
from class_function_reused import Timer
from class_function_reused import Accumulator
import math
from class_function_reused import sgd

#set params
def get_params(vocab_size,num_hiddens,device):
    num_inputs=vocab_size
    num_outputs=vocab_size

    def normal(shape):
        return torch.randn(size=shape,device=device)*0.01
    
    W_xh=normal((num_inputs,num_hiddens))
    W_hh=normal((num_hiddens,num_hiddens))
    b_h=torch.zeros(num_hiddens,device=device)
    W_hq=normal((num_hiddens,num_outputs))
    b_q=torch.zeros(num_outputs,device=device)
    params=[W_xh,W_hh,b_h,W_hq,b_q]
    for param in params:
        param.requires_grad_(True)
    return params

#set H
def init_rnn_state(batch_size,num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device=device),)   #这里返回的是tuple

def rnn(inputs,state,params):
    W_xh,W_hh,b_h,W_hq,b_q=params
    H,=state
    out_puts=[]
    for X in inputs:
        H=torch.tanh(torch.mm(X,W_xh)+torch.mm(H,W_hh)+b_h)
        Y=torch.mm(H,W_hq)+b_q
        out_puts.append(Y)
    return torch.cat(out_puts,dim=0),(H,)

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

#define prediction according to prefix    
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

#梯度裁剪
def grad_clipping(net,theta):
    if isinstance(net,nn.Module):
        params=[p for p in net.parameters() if p.requires_grad]
    else:
        params=net.params
    norm=torch.sqrt(sum(torch.sum(param.grad**2) for param in params))
    if norm>theta:
        for param in params:
            param.grad[:]*=theta/norm

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

if __name__=="__main__":
    batch_size=32
    num_steps=35
    train_iter,vocab=load_data_time_machine(batch_size,num_steps)
    num_hiddens=512
    net=RNN_Model(len(vocab),num_hiddens,try_gpu(),get_params,init_rnn_state,rnn)
    num_epochs=500
    learning_rate=1
    print("Sequential:")
    train(net,train_iter,vocab,learning_rate,num_epochs,try_gpu())
    print("Random:")
    net=RNN_Model(len(vocab),num_hiddens,try_gpu(),get_params,init_rnn_state,rnn)
    train(net,train_iter,vocab,learning_rate,num_epochs,try_gpu(),use_random_iter=True)