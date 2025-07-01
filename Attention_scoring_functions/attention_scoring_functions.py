import torch
import math
from torch import nn
from class_function_reused import sequence_mask

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

#加性注意力
class AdditiveAttention(nn.Module):
    def __init__(self,key_size,query_size,num_hiddens,dropout,**kwargs):
        super().__init__(**kwargs)
        self.W_k=nn.Linear(key_size,num_hiddens,bias=False)
        self.W_q=nn.Linear(query_size,num_hiddens,bias=False)
        self.W_v=nn.Linear(num_hiddens,1,bias=False)
        self.dropout=nn.Dropout(dropout)

    def forward(self,queries,keys,values,valid_lens):
        queries=self.W_q(queries)   #(batch_size,num_queries,num_hiddens)
        keys=self.W_k(keys) #(batch_size,num_key_value_pairs,num_hiddens)
        features=queries.unsqueeze(2)+keys.unsqueeze(1)
        features=torch.tanh(features)
        scores=self.W_v(features).squeeze(-1)
        self.attention_weights=masked_softmax(scores,valid_lens)
        #attention_weights shape:(batch_size,num_queries,num_key_value_pairs)
        #value shape:(batch_size,num_key_value_pairs,dim_values)
        return torch.bmm(self.dropout(self.attention_weights),values)

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
    
if __name__=="__main__":
    #加性注意力
    queries=torch.normal(0,1,(2,1,20))
    keys=torch.ones((2,10,2))
    values=torch.arange(40,dtype=torch.float32).reshape((1,10,4)).repeat(2,1,1)
    valid_lens=torch.tensor([2,6])
    attention=AdditiveAttention(key_size=2,query_size=20,num_hiddens=8,dropout=0.1)
    attention.eval()
    print(attention(queries,keys,values,valid_lens))

    #缩放点积注意力
    queries=torch.normal(0,1,(2,1,2))
    attention=DotProductAttention(dropout=0.5)
    attention.eval()
    print(attention(queries,keys,values,valid_lens))