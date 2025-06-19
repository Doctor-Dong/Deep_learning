import torch
from torch import nn
from dataset import load_data_time_machine
from GPU import try_gpu
from class_function_reused import RNN_Model
from class_function_reused import train

#set params
def get_LSTM_params(vocab_size,num_hiddens,device):
    num_inputs=vocab_size
    num_outputs=vocab_size

    def normal(shape):
        return torch.randn(size=shape,device=device)*0.01
    
    def three_normal():
        return (normal((num_inputs,num_hiddens)),normal((num_hiddens,num_hiddens)),torch.zeros(num_hiddens,device=device))
    
    W_xi,W_hi,b_i=three_normal()    #输入门参数
    W_xf,W_hf,b_f=three_normal()    #遗忘门参数
    W_xo,W_ho,b_o=three_normal()    #输出门参数
    W_xc,W_hc,b_c=three_normal()    #候选记忆元参数
    W_hq=normal((num_hiddens,num_outputs))
    b_q=torch.zeros(num_outputs,device=device)
    params=[W_xi,W_hi,b_i,W_xf,W_hf,b_f,W_xo,W_ho,b_o,W_xc,W_hc,b_c,W_hq,b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_LSTM_state(batch_size,num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device=device),torch.zeros((batch_size,num_hiddens),device=device))

def LSTM(inputs,state,params):
    [W_xi,W_hi,b_i,W_xf,W_hf,b_f,W_xo,W_ho,b_o,W_xc,W_hc,b_c,W_hq,b_q]=params
    (H,C)=state
    outputs=[]
    for X in inputs:
        I=torch.sigmoid(torch.matmul(X,W_xi)+torch.matmul(H,W_hi)+b_i)
        F=torch.sigmoid(torch.matmul(X,W_xf)+torch.matmul(H,W_hf)+b_f)
        O=torch.sigmoid(torch.matmul(X,W_xo)+torch.matmul(H,W_ho)+b_o)
        C_tilda=torch.tanh(torch.matmul(X,W_xc)+torch.matmul(H,W_hc)+b_c)
        C=F*C+I*C_tilda
        H=O*torch.tanh(C)
        Y=torch.matmul(H,W_hq)+b_q
        outputs.append(Y)
    return torch.cat(outputs,dim=0),(H,C)

if __name__=="__main__":
    batch_size=32
    num_steps=35
    train_iter,vocab=load_data_time_machine(batch_size,num_steps)
    vocab_size=len(vocab)
    num_hiddens=256
    device=try_gpu()
    num_epochs=500
    learning_rate=1
    net=RNN_Model(vocab_size,num_hiddens,device,get_LSTM_params,init_LSTM_state,LSTM)
    train(net,train_iter,vocab,learning_rate,num_epochs,device)