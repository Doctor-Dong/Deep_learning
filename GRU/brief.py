import torch
from torch import nn
from class_function_reused import train
from class_function_reused import RNNModel
from dataset import load_data_time_machine
from GPU import try_gpu

if __name__=="__main__":
    batch_size=32
    num_steps=35
    train_iter,vocab=load_data_time_machine(batch_size,num_steps)
    vocab_size=len(vocab)
    num_hiddens=256
    device=try_gpu()
    num_epochs=500
    learning_rate=1
    num_inputs=vocab_size
    gru_layer=nn.GRU(num_inputs,num_hiddens)
    net=RNNModel(gru_layer,vocab_size)
    net=net.to(device)
    train(net,train_iter,vocab,learning_rate,num_epochs,device)