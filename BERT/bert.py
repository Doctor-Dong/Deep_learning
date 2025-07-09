import torch
from torch import nn
from class_function_reused import EncoderBlock

def get_tokens_and_segments(tokens_a,tokens_b=None):
    tokens=["<cls>"]+tokens_a+["<sep>"]
    segments=[0]*(len(tokens_a)+2)  #0和1分别标记片段A和B
    if tokens_b is not None:
        tokens+=tokens_b+["<sep>"]
        segments+=[1]*(len(tokens_b)+1)
    return tokens,segments

class BERTEncoder(nn.Module):
    def __init__(self,vocab_size,num_hiddens,normalized_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,num_layers,dropout,max_len=1000,key_size=768,query_size=768,value_size=768,**kwargs):
        super().__init__(**kwargs)
        self.token_embedding=nn.Embedding(vocab_size,num_hiddens)   #词元嵌入
        self.segment_embedding=nn.Embedding(2,num_hiddens)  #片段嵌入
        self.blks=nn.Sequential()
        for i in range(0,num_layers):
            self.blks.add_module(f"{i}",EncoderBlock(key_size,query_size,value_size,num_hiddens,normalized_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,dropout,True))
        self.pos_embedding=nn.Parameter(torch.randn((1,max_len,num_hiddens)))   #位置嵌入

    def forward(self,tokens,segments,valid_len):
        #X shape:(batch_size,最大序列长度,num_hiddens)
        X=self.token_embedding(tokens)+self.segment_embedding(segments)
        X=X+self.pos_embedding[:,:X.shape[1],:] #broadcast
        for blk in self.blks:
            X=blk(X,valid_len)
        return X

#掩蔽语言模型(Masked Language Modeling)
class MaskLM(nn.Module):
    def __init__(self,vocab_size,num_hiddens,num_inputs=768,**kwargs):
        super().__init__(**kwargs)
        self.mlp=nn.Sequential(nn.Linear(num_inputs,num_hiddens),nn.ReLU(),nn.LayerNorm(num_hiddens),nn.Linear(num_hiddens,vocab_size))

    def forward(self,X,pred_positions):
        #X shape:(batch_size,seq_len,num_inputs)
        #pred_positions shape:(batch_size,num_pred_positions)
        num_pred_positions=pred_positions.shape[1]
        pred_positions=pred_positions.reshape(-1)
        batch_size=X.shape[0]
        batch_idx=torch.arange(0,batch_size)
        batch_idx=torch.repeat_interleave(batch_idx,num_pred_positions)
        masked_X=X[batch_idx,pred_positions]
        masked_X=masked_X.reshape((batch_size,num_pred_positions,-1))
        mlm_Y_hat=self.mlp(masked_X)
        return mlm_Y_hat
    
#下一句预测
class NextSentencePred(nn.Module):
    def __init__(self,num_inputs,**kwargs):
        super().__init__(**kwargs)
        self.output=nn.Linear(num_inputs,2)
    
    def forward(self,X):
        #X shape:(batch_size,num_hiddens)，是编码后的"<cls>"词元
        return self.output(X)

#整合代码
class BERTModel(nn.Module):
    def __init__(self,vocab_size,num_hiddens,normalized_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,num_layers,dropout,
                 max_len=1000,key_size=768,query_size=768,value_size=768,hid_in_features=768,mlm_in_features=768,nsp_in_features=768):
        super().__init__()
        self.encoder=BERTEncoder(vocab_size,num_hiddens,normalized_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,num_layers,dropout,max_len=max_len,key_size=key_size,query_size=query_size,value_size=value_size)
        self.hidden=nn.Sequential(nn.Linear(hid_in_features,num_hiddens),nn.Tanh())
        self.mlm=MaskLM(vocab_size,num_hiddens,mlm_in_features)
        self.nsp=NextSentencePred(nsp_in_features)
    
    def forward(self,tokens,segments,valid_lens=None,pred_positions=None):
        encoded_X=self.encoder(tokens,segments,valid_lens)
        if pred_positions is not None:
            mlm_Y_hat=self.mlm(encoded_X,pred_positions)
        else:
            mlm_Y_hat=None
        nsp_Y_hat=self.nsp(self.hidden(encoded_X[:,0,:]))
        return encoded_X,mlm_Y_hat,nsp_Y_hat

if __name__=="__main__":
    vocab_size=10000
    num_hiddens=768
    ffn_num_inputs=768
    ffn_num_hiddens=1024
    num_heads=4
    normalized_shape=[768]
    num_layers=2
    dropout=0.2
    encoder=BERTEncoder(vocab_size,num_hiddens,normalized_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,num_layers,dropout)

    tokens=torch.randint(0,vocab_size,(2,8))
    segments=torch.tensor([[0,0,0,0,1,1,1,1],[0,0,0,1,1,1,1,1]])
    encoded_X=encoder(tokens,segments,None)
    print(encoded_X.shape)

    mlm=MaskLM(vocab_size,num_hiddens)
    mlm_positions=torch.tensor([[1,5,2],[6,1,5]])
    mlm_Y_hat=mlm(encoded_X,mlm_positions)
    print(mlm_Y_hat.shape)
    mlm_Y=torch.tensor([[7,8,9],[10,20,30]])
    loss=nn.CrossEntropyLoss(reduction="none")
    mlm_loss=loss(mlm_Y_hat.reshape((-1,vocab_size)),mlm_Y.reshape(-1))
    print(mlm_loss.shape)

    encoded_cls=encoded_X[:,0,:]
    nsp=NextSentencePred(encoded_cls.shape[-1])
    nsp_Y_hat=nsp(encoded_cls)
    print(nsp_Y_hat.shape)
    nsp_y=torch.tensor([0,1])
    nsp_loss=loss(nsp_Y_hat,nsp_y)
    print(nsp_loss.shape)

    net=BERTModel(vocab_size,num_hiddens,normalized_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,num_layers,dropout)