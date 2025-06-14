import torch
from torch import nn
from dataset import load_data_time_machine
from torch.nn import functional
from GPU import try_gpu
from from_zero import train

class RNNModel(nn.Module):
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

if __name__=="__main__":
    batch_size=32
    num_steps=35
    train_iter,vocab=load_data_time_machine(batch_size,num_steps)
    num_hiddens=256
    rnn_layer=nn.RNN(len(vocab),num_hiddens)
    #state的形状为(隐藏层数，批量大小，隐藏单元数)
    device=try_gpu()
    net=RNNModel(rnn_layer,len(vocab))
    net=net.to(device)
    num_epochs=500
    learning_rate=1
    train(net,train_iter,vocab,learning_rate,num_epochs,device)