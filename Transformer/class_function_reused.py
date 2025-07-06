import torch
from torch import nn
import math
from encoder_decoder import Decoder
from collections import Counter
from torch.utils import data
import numpy
import time
import collections

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
    
class AttentionDecoder(Decoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError
    
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
    
def load_array(data_arrays,batch_size,is_train=True):
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

def grad_clipping(net,theta):
    if isinstance(net,nn.Module):
        params=[p for p in net.parameters() if p.requires_grad]
    else:
        params=net.params
    norm=torch.sqrt(sum(torch.sum(param.grad**2) for param in params))
    if norm>theta:
        for param in params:
            param.grad[:]*=theta/norm

def sequence_mask(X,valid_len,value=0):
    maxlen=X.size(1)
    mask=torch.arange(maxlen,dtype=torch.float32,device=X.device)[None,:]<valid_len[:,None]
    X[~mask]=value
    return X

#define loss
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self,pred,label,valid_len):
        #pred shape:(batch_size,num_steps,vocab_size)
        #label shape:(batch_size,num_steps)
        #valid_len shape:(batchsize,)
        weights=torch.ones_like(label)
        weights=sequence_mask(weights,valid_len)
        self.reduction="none"
        unweighted_loss=super(MaskedSoftmaxCELoss,self).forward(pred.permute(0,2,1),label)
        weighted_loss=(unweighted_loss*weights).mean(dim=1)
        return weighted_loss

#define train 
def train_seq2seq(net,data_iter,learning_rate,num_epochs,target_vocab,device):
    def xavier_init_weights(m):
        if type(m)==nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m)==nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate)
    loss=MaskedSoftmaxCELoss()
    net.train()
    for epoch in range(0,num_epochs):
        timer=Timer()
        metric=Accumulator(2)
        timer.start()
        for batch in data_iter:
            optimizer.zero_grad()
            X,X_valid_len,Y,Y_valid_len=[x.to(device) for x in batch]
            bos=torch.tensor([target_vocab["<bos>"]]*Y.shape[0],device=device).reshape(-1,1)
            dec_input=torch.cat((bos,Y[:,:-1]),dim=1)
            Y_hat,_=net(X,dec_input,X_valid_len)
            temp_loss=loss(Y_hat,Y,Y_valid_len)
            temp_loss.sum().backward()
            grad_clipping(net,1)
            num_tokens=Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(temp_loss.sum(),num_tokens)
        if (epoch+1)%10==0:
            print(f"epoch{epoch+1}:loss={metric[0]/metric[1]:.3f}")
    print(f"loss={metric[0]/metric[1]:.3f},{metric[1]/timer.stop():.1f} tokens/second on {str(device)}")

def truncate_pad(line,num_steps,padding_token):
    if len(line)>num_steps:
        return line[:num_steps]
    return line+[padding_token]*(num_steps-len(line))

#define predict
def predict_seq2seq(net,source_sentence,source_vocab,target_vocab,num_steps,device,save_attention_weights=False):
    net.eval()
    source_tokens=source_vocab[source_sentence.lower().split(" ")]+[source_vocab["<eos>"]]
    enc_valid_len=torch.tensor([len(source_tokens)],device=device)
    source_tokens=truncate_pad(source_tokens,num_steps,padding_token=source_vocab["<pad>"])
    enc_X=torch.unsqueeze(torch.tensor(source_tokens,dtype=torch.long,device=device),dim=0)
    enc_outputs=net.encoder(enc_X,enc_valid_len)
    dec_state=net.decoder.init_state(enc_outputs,enc_valid_len)
    dec_X=torch.unsqueeze(torch.tensor([target_vocab["<bos>"]],dtype=torch.long,device=device),dim=0)
    output_seq=[]
    attention_weight_seq=[]
    for _ in range(0,num_steps):
        Y,dec_state=net.decoder(dec_X,dec_state)
        dec_X=Y.argmax(dim=2)
        pred=dec_X.squeeze(dim=0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred==target_vocab["<eos>"]:
            break
        output_seq.append(pred)
    return " ".join(target_vocab.to_tokens(output_seq)),attention_weight_seq

def bleu(pred_seq,label_seq,k):
    pred_tokens=pred_seq.split(" ")
    label_tokens=label_seq.split(" ")
    len_pred=len(pred_tokens)
    len_label=len(label_tokens)
    score=math.exp(min(0,1-len_label/len_pred))
    for n in range(1,k+1):
        num_matches=0
        label_subs=collections.defaultdict(int)
        for i in range(0,len_label-n+1):
            label_subs[" ".join(label_tokens[i:i+n])]+=1
        for i in range(0,len_pred-n+1):
            if label_subs[" ".join(label_tokens[i:i+n])]>0:
                num_matches+=1
                label_subs[" ".join(label_tokens[i:i+n])]-=1
        score*=math.pow(num_matches/(len_pred-n+1),math.pow(0.5,n))
    return score