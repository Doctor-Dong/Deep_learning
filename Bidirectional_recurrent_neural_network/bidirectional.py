import torch
from torch import nn
from GPU import try_gpu
from dataset import load_data_time_machine
from class_function_reused import train
from class_function_reused import RNNModel

if __name__=="__main__":
    batch_size=32
    num_steps=35
    device=try_gpu()
    train_iter,vocab=load_data_time_machine(batch_size,num_steps)
    vocab_size=len(vocab)
    num_hiddens=256
    num_layers=2
    num_inputs=vocab_size
    num_outputs=vocab_size
    LSTM_layer=nn.LSTM(num_inputs,num_hiddens,num_layers,bidirectional=True)
    net=RNNModel(LSTM_layer,num_outputs)
    net=net.to(device)
    num_epochs=500
    learning_rate=1
    train(net,train_iter,vocab,learning_rate,num_epochs,device)