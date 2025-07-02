import torch
import math
from torch import nn

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