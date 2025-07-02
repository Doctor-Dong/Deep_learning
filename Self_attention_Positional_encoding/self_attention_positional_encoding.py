import math
import torch
from torch import nn
from class_function_reused import Multi_Head_Attention

class PositionalEncoding(nn.Module):
    def __init__(self,num_hiddens,dropout,max_len=1000):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.P=torch.zeros((1,max_len,num_hiddens))
        X=torch.arange(max_len,dtype=torch.float32).reshape(-1,1)/torch.pow(10000,torch.arange(0,num_hiddens,2,dtype=torch.float32)/num_hiddens)
        self.P[:,:,0::2]=torch.sin(X)
        self.P[:,:,1::2]=torch.cos(X)
    
    def forward(self,X):
        X=X+self.P[:,0:X.shape[1],:].to(X.device)
        return self.dropout(X)

if __name__=="__main__":
    num_hiddens=100
    num_heads=5
    attention=Multi_Head_Attention(num_hiddens,num_hiddens,num_hiddens,num_hiddens,num_heads,dropout=0.5)
    attention.eval()
    batch_size=2
    num_queries=4
    valid_lens=torch.tensor([3,2])
    X=torch.ones((batch_size,num_queries,num_hiddens))
    print(attention(X,X,X,valid_lens).shape)    #自注意力

    #位置编码
    encoding_dim=32
    num_steps=60
    pos_encoding=PositionalEncoding(encoding_dim,0)
    pos_encoding.eval()
    X=pos_encoding(torch.zeros((1,num_steps,encoding_dim)))
    print(X)
    P=pos_encoding.P[:,0:X.shape[1],:]
    print(P)