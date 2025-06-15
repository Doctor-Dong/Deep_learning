import torch
from torch import nn
from dataset import load_data_time_machine
from GPU import try_gpu
from class_function_reused import RNN_Model
from class_function_reused import train

#set params
def get_params(vocab_size,num_hiddens,device):
    num_inputs=vocab_size
    num_outputs=vocab_size

    def normal(shape):
        return torch.randn(size=shape,device=device)*0.01
    
    def three_normal():
        return (normal((num_inputs,num_hiddens)),normal((num_hiddens,num_hiddens)),torch.zeros(num_hiddens,device=device))
    
    W_xz,W_hz,b_z=three_normal()    #更新门参数
    W_xr,W_hr,b_r=three_normal()    #重置门参数
    W_xh,W_hh,b_h=three_normal()    #候选隐状态参数
    W_hq=normal((num_hiddens,num_outputs))
    b_q=torch.zeros(num_outputs,device=device)
    params=[W_xz,W_hz,b_z,W_xr,W_hr,b_r,W_xh,W_hh,b_h,W_hq,b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size,num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device=device),)

def gru(inputs,state,params):
    W_xz,W_hz,b_z,W_xr,W_hr,b_r,W_xh,W_hh,b_h,W_hq,b_q=params
    H,=state
    outputs=[]
    for X in inputs:
        Z=torch.sigmoid(torch.matmul(X,W_xz)+torch.matmul(H,W_hz)+b_z)
        R=torch.sigmoid(torch.matmul(X,W_xr)+torch.matmul(H,W_hr)+b_r)
        H_tilda=torch.tanh(torch.matmul(X,W_xh)+torch.matmul((R*H),W_hh)+b_h)
        H=Z*H+(1-Z)*H_tilda
        Y=torch.matmul(H,W_hq)+b_q
        outputs.append(Y)
    return torch.cat(outputs,dim=0),(H,)

if __name__=="__main__":
    batch_size=32
    num_steps=35
    train_iter,vocab=load_data_time_machine(batch_size,num_steps)
    vocab_size=len(vocab)
    num_hiddens=256
    device=try_gpu()
    num_epochs=500
    learning_rate=1
    net=RNN_Model(vocab_size,num_hiddens,device,get_params,init_gru_state,gru)
    train(net,train_iter,vocab,learning_rate,num_epochs,device)
