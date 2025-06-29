import torch
from torch import nn
from dataset import load_data_time_machine
from GPU import try_gpu
from class_function_reused import RNNModel
from class_function_reused import train

if __name__=="__main__":
    batch_size=32
    num_steps=35
    train_iter,vocab=load_data_time_machine(batch_size,num_steps)
    vocab_size=len(vocab)
    num_hiddens=256
    num_layers=2
    num_inputs=vocab_size
    num_outputs=vocab_size
    device=try_gpu()
    LSTM_layer=nn.LSTM(num_inputs,num_hiddens,num_layers)
    net=RNNModel(LSTM_layer,vocab_size)
    net=net.to(device)
    num_epochs=500
    learning_rate=2
    train(net,train_iter,vocab,learning_rate,num_epochs,device)