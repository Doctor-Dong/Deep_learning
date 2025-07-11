from collections import Counter
import torch
from torch import nn
import math
import time
import numpy

class Timer:
    def __init__(self):
        self.times=[]

    def start(self):
        self.start_time=time.time()

    def stop(self): #记录并返回这段时长
        self.times.append(time.time()-self.start_time)
        return self.times[-1]
    
    def average(self):
        return sum(self.times)/len(self.times)
    
    def total_time(self):
        return sum(self.times)
    
    def cumsum_time(self):
        return numpy.array(self.times).cumsum().tolist()
    
class Accumulator:
    def __init__(self,n):
        self.data=[0.0]*n

    def __getitem__(self,index):
        return self.data[index]
    
    def add(self,*params):
        self.data=[a+float(b) for a,b in zip(self.data,params) ]

    def reset(self):
        self.data=[0.0]*len(self.data)

def get_tokens_and_segments(tokens_a,tokens_b=None):
    tokens=["<cls>"]+tokens_a+["<sep>"]
    segments=[0]*(len(tokens_a)+2)  #0和1分别标记片段A和B
    if tokens_b is not None:
        tokens+=tokens_b+["<sep>"]
        segments+=[1]*(len(tokens_b)+1)
    return tokens,segments

#define tokenize
def tokenize(lines,token="word"):
    if token=="word":
        return [line.split() for line in lines]
    elif token=="char":
        return [list(line) for line in lines]
    else:
        print("错误：未知词元类型："+token)

def count_freq(tokens):
    if len(tokens)==0 or isinstance(tokens[0],list):
        tokens=[token for line in tokens for token in line]
    return Counter(tokens)

class Vocab:
    def __init__(self,tokens=None,reserved_tokens=None,min_freq=0):
        if tokens is None:
            tokens=[]
        if reserved_tokens is None:
            reserved_tokens=[]
        counter=count_freq(tokens)
        self.token_freqs=sorted(counter.items(),key=lambda x:x[1],reverse=True)
        #self.token_freq是一个list，里面的每个元素是个有两个元素的tuple
        self.index_to_token=["<unk>"]+reserved_tokens
        self.token_to_index={token:index for index,token in enumerate(self.index_to_token)}
        for token,freq in self.token_freqs:
            if freq<min_freq:
                break
            if token not in self.token_to_index:
                self.index_to_token.append(token)
                self.token_to_index[token]=len(self.index_to_token)-1
    
    def __len__(self):
        return len(self.index_to_token)
    
    def __getitem__(self,tokens):
        if not isinstance(tokens,(tuple,list)):
            return self.token_to_index.get(tokens,self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self,indices):
        if not isinstance(indices,(tuple,list)):
            return self.index_to_token[indices]
        return [self.index_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0
    
    @property
    def token_freq(self):
        return self.token_freqs

def get_dataloader_workers():
    return 0

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

#基于位置的前馈网络
class PositionWiseFFN(nn.Module):
    def __init__(self,ffn_num_inputs,ffn_num_hiddens,ffn_num_outputs,**kwargs):
        super().__init__(**kwargs)
        self.dense1=nn.Linear(ffn_num_inputs,ffn_num_hiddens)
        self.relu=nn.ReLU()
        self.dense2=nn.Linear(ffn_num_hiddens,ffn_num_outputs)
    
    def forward(self,X):
        return self.dense2(self.relu(self.dense1(X)))
    
#残差连接和层规范化
class AddNorm(nn.Module):
    def __init__(self,normalized_shape,dropout,**kwargs):
        super().__init__(**kwargs)
        self.dropout=nn.Dropout(dropout)
        self.ln=nn.LayerNorm(normalized_shape)
    
    def forward(self,X,Y):
        return self.ln(self.dropout(Y)+X)

class EncoderBlock(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,normalized_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,dropout,use_bias=False,**kwargs):
        super().__init__(**kwargs)
        self.attention=Multi_Head_Attention(key_size,query_size,value_size,num_hiddens,num_heads,dropout,use_bias)
        self.addnorm1=AddNorm(normalized_shape,dropout)
        self.fnn=PositionWiseFFN(ffn_num_inputs,ffn_num_hiddens,num_hiddens)
        self.addnorm2=AddNorm(normalized_shape,dropout)
    
    def forward(self,X,valid_lens):
        Y=self.addnorm1(X,self.attention(X,X,X,valid_lens))
        return self.addnorm2(Y,self.fnn(Y))