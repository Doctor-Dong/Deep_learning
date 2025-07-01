import torch
from torch import nn
from encoder_decoder import Decoder
from class_function_reused import AdditiveAttention
from GPU import try_gpu
from dataset import load_data_nmt
from class_function_reused import Seq2SeqEncoder
from encoder_decoder import Encoder_Decoder
from class_function_reused import train_seq2seq
from class_function_reused import bleu
from class_function_reused import predict_seq2seq

class AttentionDecoder(Decoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0,**kwargs):
        super().__init__(**kwargs)
        self.attention=AdditiveAttention(num_hiddens,num_hiddens,num_hiddens,dropout)
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.rnn=nn.GRU(embed_size+num_hiddens,num_hiddens,num_layers,dropout=dropout)
        self.dense=nn.Linear(num_hiddens,vocab_size)

    def init_state(self,enc_outputs,enc_valid_lens,*args):
        outputs,hidden_state=enc_outputs
        return (outputs.permute(1,0,2),hidden_state,enc_valid_lens)
        #return shape:  outputs:(batch_size,num_steps,num_hiddens) , hidden_state:(num_layers,batch_size,num_hiddens)
    
    def forward(self,X,state):
        enc_outputs,hidden_state,enc_valid_lens=state
        X=self.embedding(X).permute(1,0,2)  #(num_steps,batch_size,embed_size)
        outputs=[]
        self._attention_weights=[]
        for x in X:
            query=torch.unsqueeze(hidden_state[-1],dim=1)   #(batch_size,1,num_hiddens)<-num_queries=1
            context=self.attention(query,enc_outputs,enc_outputs,enc_valid_lens)
            #context shape:(batch_size,num_queries,num_values)=(batch_size,1,num_hiddens)
            x=torch.cat((context,torch.unsqueeze(x,dim=1)),dim=-1)  #(batch_size,1,num_hiddens+embed_size)
            out,hidden_state=self.rnn(x.permute(1,0,2),hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        outputs=self.dense(torch.cat(outputs,dim=0))    #(num_steps,batch_size,vocab_size)
        return outputs.permute(1,0,2),[enc_outputs,hidden_state,enc_valid_lens]
    
    @property
    def attention_weights(self):
        return self._attention_weights

if __name__=="__main__":
    embed_size=32
    num_hiddens=32
    num_layers=2
    dropout=0.1
    batch_size=64
    num_steps=10
    learning_rate=0.005
    num_epochs=250
    device=try_gpu()
    train_iter,source_vocab,target_vocab=load_data_nmt(batch_size,num_steps)
    encoder=Seq2SeqEncoder(len(source_vocab),embed_size,num_hiddens,num_layers,dropout)
    decoder=Seq2SeqAttentionDecoder(len(target_vocab),embed_size,num_hiddens,num_layers,dropout)
    net=Encoder_Decoder(encoder,decoder)
    train_seq2seq(net,train_iter,learning_rate,num_epochs,target_vocab,device)

    engs=["go .","i lost .","he\'s calm .","i\'m home ."]
    fras=["va !","j\'ai perdu .","il est calme .","je suis chez moi ."]
    for eng,fra in zip(engs,fras):
        translation,dec_attention_weight_seq=predict_seq2seq(net,eng,source_vocab,target_vocab,num_steps,device,True)
        print(f"{eng}=>{translation},bleu={bleu(translation,fra,k=2):.3f}")