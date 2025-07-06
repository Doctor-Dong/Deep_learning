import torch
import math
from torch import nn
from class_function_reused import Multi_Head_Attention
from encoder_decoder import Encoder
from class_function_reused import PositionalEncoding
from class_function_reused import AttentionDecoder
from GPU import try_gpu
from dataset import load_data_nmt
from encoder_decoder import Encoder_Decoder
from class_function_reused import train_seq2seq
from class_function_reused import predict_seq2seq
from class_function_reused import bleu

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
    
#编码器
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
    
class TransformerEncoder(Encoder):
    def __init__(self,vocab_size,key_size,query_size,value_size,num_hiddens,normalized_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,num_layers,dropout,use_bias=False,**kwargs):
        super().__init__(**kwargs)
        self.num_hiddens=num_hiddens
        self.embedding=nn.Embedding(vocab_size,num_hiddens)
        self.pos_encoding=PositionalEncoding(num_hiddens,dropout)
        self.blks=nn.Sequential()
        for i in range(0,num_layers):
            self.blks.add_module("block"+str(i),EncoderBlock(key_size,query_size,value_size,num_hiddens,normalized_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,dropout,use_bias))

    def forward(self,X,valid_lens,*args):
        X=self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self.attention_weights=[None]*len(self.blks)
        for i,blk in enumerate(self.blks):
            X=blk(X,valid_lens)
            self.attention_weights[i]=blk.attention.attention.attention_weights
        return X
    
#解码器
class DecoderBlock(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,normalized_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,dropout,i,**kwargs):
        super().__init__(**kwargs)
        self.i=i
        self.attention1=Multi_Head_Attention(key_size,query_size,value_size,num_hiddens,num_heads,dropout)
        self.addnorm1=AddNorm(normalized_shape,dropout)
        self.attention2=Multi_Head_Attention(key_size,query_size,value_size,num_hiddens,num_heads,dropout)
        self.addnorm2=AddNorm(normalized_shape,dropout)
        self.ffn=PositionWiseFFN(ffn_num_inputs,ffn_num_hiddens,num_hiddens)
        self.addnorm3=AddNorm(normalized_shape,dropout)
        
    def forward(self,X,state):
        enc_outputs=state[0]
        enc_valid_lens=state[1]
        if state[2][self.i] is None:
            key_values=X
        else:
            key_values=torch.cat((state[2][self.i],X),axis=1)
        state[2][self.i]=key_values
        if self.training:
            batch_size,num_steps,_=X.shape
            dec_valid_lens=torch.arange(1,num_steps+1,device=X.device).repeat(batch_size,1)
        else:
            dec_valid_lens=None
        X2=self.attention1(X,key_values,key_values,dec_valid_lens)
        Y=self.addnorm1(X,X2)
        Y2=self.attention2(Y,enc_outputs,enc_outputs,enc_valid_lens)
        Z=self.addnorm2(Y,Y2)
        return self.addnorm3(Z,self.ffn(Z)),state

class TransformerDecoder(AttentionDecoder):
    def __init__(self,vocab_size,key_size,query_size,value_size,num_hiddens,normalized_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,num_layers,dropout,**kwargs):
        super().__init__(**kwargs)
        self.num_hiddens=num_hiddens
        self.num_layers=num_layers
        self.embedding=nn.Embedding(vocab_size,num_hiddens)
        self.pos_encoding=PositionalEncoding(num_hiddens,dropout)
        self.blks=nn.Sequential()
        for i in range(0,num_layers):
            self.blks.add_module("block"+str(i),DecoderBlock(key_size,query_size,value_size,num_hiddens,normalized_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,dropout,i))
        self.dense=nn.Linear(num_hiddens,vocab_size)
    
    def init_state(self,enc_outputs,enc_valid_lens,*args):
        return [enc_outputs,enc_valid_lens,[None]*self.num_layers]

    def forward(self,X,state):
        X=self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self._attention_weights=[[None]*len(self.blks) for _ in range(0,2)]
        for i,blk in enumerate(self.blks):
            X,state=blk(X,state)
            self._attention_weights[0][i]=blk.attention1.attention.attention_weights
            self._attention_weights[1][i]=blk.attention2.attention.attention_weights
        return self.dense(X),state
    
    @property
    def attention_weights(self):
        return self._attention_weights
    
if __name__=="__main__":
    num_hiddens=32
    num_layers=2
    dropout=0.1
    batch_size=64
    num_steps=10
    learning_rate=0.005
    num_epochs=200
    device=try_gpu()
    ffn_num_inputs=32
    ffn_num_hiddens=64
    num_heads=4
    key_size=32
    query_size=32
    value_size=32
    normalized_shape=[32]
    train_iter,source_vocab,target_vocab=load_data_nmt(batch_size,num_steps)
    encoder=TransformerEncoder(len(source_vocab),key_size,query_size,value_size,num_hiddens,normalized_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,num_layers,dropout)
    decoder=TransformerDecoder(len(target_vocab),key_size,query_size,value_size,num_hiddens,normalized_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,num_layers,dropout)
    net=Encoder_Decoder(encoder,decoder)
    train_seq2seq(net,train_iter,learning_rate,num_epochs,target_vocab,device)

    engs=["go .","i lost .","he\'s calm .","i\'m home ."]
    fras=["va !","j\'ai perdu .","il est calme .","je suis chez moi ."]
    for eng,fra in zip(engs,fras):
        translation,dec_attention_weight_seq=predict_seq2seq(net,eng,source_vocab,target_vocab,num_steps,device,True)
        print(f"{eng}=>{translation},bleu={bleu(translation,fra,k=2):.3f}")