import torch
from torch import nn
import math

def sequence_mask(X,valid_len,value=0):
    maxlen=X.size(1)
    mask=torch.arange(maxlen,dtype=torch.float32,device=X.device)[None,:]<valid_len[:,None]
    X[~mask]=value
    return X

def masked_softmax(X,valid_lens):
    #X shape:(batch_size,num_queries,seq_len)
    #valid_lens shape:(batch_size)/(batch_size,num_quries)
    if valid_lens is None:
        return nn.functional.softmax(X,dim=-1)
    else:
        shape=X.shape
        if valid_lens.dim()==1: #每个batch的valid_lens相同
            valid_lens=torch.repeat_interleave(valid_lens,shape[1])
        else:   #每个batch的每个query有自己的valid_lens
            valid_lens=valid_lens.reshape(-1)
        X=sequence_mask(X.reshape((-1,shape[-1])),valid_lens,value=-1e6)
        return nn.functional.softmax(X.reshape(shape),dim=-1)

#缩放点积注意力（query和key要有相同的长度d）
class DotProductAttention(nn.Module):
    def __init__(self,dropout,**kwargs):
        super().__init__(**kwargs)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,queries,keys,values,valid_lens=None):
        d=queries.shape[-1]
        scores=torch.bmm(queries,keys.transpose(1,2))/math.sqrt(d)
        self.attention_weights=masked_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(self.attention_weights),values)

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