import torch
import math
import collections
from torch import nn
from encoder_decoder import Encoder
from encoder_decoder import Decoder
from class_function_reused import Timer
from class_function_reused import Accumulator
from class_function_reused import grad_clipping
from GPU import try_gpu
from dataset import load_data_nmt
from encoder_decoder import Encoder_Decoder
from dataset import truncate_pad

class Seq2SeqEncoder(Encoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0,**kwargs):
        super().__init__(**kwargs)
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.rnn=nn.GRU(embed_size,num_hiddens,num_layers,dropout=dropout)
    
    def forward(self, X, *args):
        X=self.embedding(X) #(batch_size,num_steps,embed_size)
        X=X.permute(1,0,2)  #(num_steps,batch_size,embed_size)
        output,state=self.rnn(X)
        return output,state
    
class Seq2SeqDecoder(Decoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0,**kwargs):
        super().__init__(**kwargs)
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.rnn=nn.GRU(embed_size+num_hiddens,num_hiddens,num_layers,dropout=dropout)
        self.dense=nn.Linear(num_hiddens,vocab_size)
    
    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]   #enc_outputs=(output,state)
    
    def forward(self, X, state):
        X=self.embedding(X).permute(1,0,2)  #(num_steps,batch_size,embed_size)
        context=state[-1].repeat(X.shape[0],1,1)
        X_context=torch.cat((X,context),dim=2)  #(num_steps,batch_size,num_hiddens+embed_size)
        output,state=self.rnn(X_context,state)
        #output shape:(num_steps,batch_size,num_hiddens)
        #state shape:(num_layers,batch_size,num_hiddens)
        output=self.dense(output).permute(1,0,2)    #(batch_size,num_steps,vocab_size)
        return output,state

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

if __name__=="__main__":
    embed_size=32
    num_hiddens=32
    num_layers=2
    dropout=0.1
    batch_size=64
    num_steps=10
    learning_rate=0.005
    num_epochs=300
    device=try_gpu()
    train_iter,source_vocab,target_vocab=load_data_nmt(batch_size,num_steps)
    encoder=Seq2SeqEncoder(len(source_vocab),embed_size,num_hiddens,num_layers,dropout=dropout)
    decoder=Seq2SeqDecoder(len(target_vocab),embed_size,num_hiddens,num_layers,dropout=dropout)
    net=Encoder_Decoder(encoder,decoder)
    train_seq2seq(net,train_iter,learning_rate,num_epochs,target_vocab,device)
    engs=["go .","i lost .","he\'s calm .","i\'m home ."]
    fras=["va !","j\'ai perdu .","il est calme .","je suis chez moi ."]
    for eng,fra in zip(engs,fras):
        translation,attention_weight_seq=predict_seq2seq(net,eng,source_vocab,target_vocab,num_steps,device)
        print(f"{eng}=>{translation},bleu={bleu(translation,fra,k=2):.3f}")