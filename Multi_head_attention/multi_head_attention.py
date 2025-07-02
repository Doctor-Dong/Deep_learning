import torch
import math
from torch import nn
from class_function_reused import DotProductAttention

def transpose_qkv(X,num_heads):
    #X shape:(batch_size,number of queries/k-v pairs,num_hiddens)
    X=X.reshape(X.shape[0],X.shape[1],num_heads,-1)
    X=X.permute(0,2,1,3)    #(batch_size,num_heads,number of queries/k-v pairs,num_hiddens/num_heads)
    return X.reshape(-1,X.shape[2],X.shape[3])  #(batch_size*num_heads,number of queries/k-v pairs,num_hiddens/num_heads)

def transpose_output(X,num_heads):  #逆转transpose_qkv的操作
    X=X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X=X.permute(0,2,1,3)
    return X.reshape(X.shape[0],X.shape[1],-1)

class Multi_Head_Attention(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,dropout,bias=False,**kwargs):
        super().__init__(**kwargs)
        self.num_heads=num_heads
        self.attention=DotProductAttention(dropout=dropout)
        self.W_q=nn.Linear(query_size,num_hiddens,bias=bias)
        self.W_k=nn.Linear(key_size,num_hiddens,bias=bias)
        self.W_v=nn.Linear(value_size,num_hiddens,bias=bias)
        self.W_o=nn.Linear(num_hiddens,num_hiddens,bias=bias)

    def forward(self,queries,keys,values,valid_lens):
        #queries/keys/values shape:(batch_size,number of queries/k-v pairs,num_hiddens)
        #valid_lens shape:(batch_size,) or (batch_size,number of queries)
        queries=transpose_qkv(self.W_q(queries),self.num_heads)
        keys=transpose_qkv(self.W_k(keys),self.num_heads)
        values=transpose_qkv(self.W_v(values),self.num_heads)
        #(batch_size*num_heads,number of queries/k-v pairs,num_hiddens/num_heads)
        if valid_lens is not None:
            valid_lens=torch.repeat_interleave(valid_lens,repeats=self.num_heads,dim=0)
        output=self.attention(queries,keys,values,valid_lens)   #(batch_size*num_heads,number of queries,num_values=num_hiddens/num_heads)
        output_concat=transpose_output(output,self.num_heads)   #(batch_size,number of queries,num_hiddens)
        return self.W_o(output_concat)

if __name__=="__main__":
    num_hiddens=100
    num_heads=5
    attention=Multi_Head_Attention(num_hiddens,num_hiddens,num_hiddens,num_hiddens,num_heads,0.5)
    attention.eval()
    batch_size=2
    num_queries=4
    num_kvpairs=6
    valid_lens=torch.tensor([3,2])
    X=torch.ones((batch_size,num_queries,num_hiddens))
    Y=torch.ones((batch_size,num_kvpairs,num_hiddens))
    print(attention(X,Y,Y,valid_lens).shape)