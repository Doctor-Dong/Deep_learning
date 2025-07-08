import torch
from torch import nn
from dataset import load_data_ptb
from GPU import try_gpu
from class_function_reused import Accumulator
from class_function_reused import Timer

def skip_gram(center,contexts_and_negatives,embed_v,embed_u):
    #center shape:(batch_size,1)
    #contexts_and_negatives shape:(batch_size,,max_len)
    v=embed_v(center)
    u=embed_u(contexts_and_negatives)
    pred=torch.bmm(v,u.permute(0,2,1))  #(batch_size,1,max_len)
    return pred

#二元交叉熵损失
class SigmoidBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,inputs,target,mask=None):  #target即为label，先前构造用来表示正例/负例
        out=nn.functional.binary_cross_entropy_with_logits(inputs,target,weight=mask,reduction="none")
        return out.mean(dim=1)

def train(net,data_iter,learning_rate,num_epochs,device=try_gpu()):
    def init_weights(m):
        if type(m)==nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net=net.to(device)
    updater=torch.optim.Adam(net.parameters(),lr=learning_rate)
    loss=SigmoidBCELoss()
    metric=Accumulator(2)
    timer=Timer()
    timer.start()
    for epoch in range(0,num_epochs):
        num_batches=len(data_iter)
        for i,batch in enumerate(data_iter):
            updater.zero_grad()
            center,context_negative,mask,label=[data.to(device) for data in batch]
            pred=skip_gram(center,context_negative,net[0],net[1])
            temp_loss=(loss(pred.reshape(label.shape).float(),label.float(),mask))*mask.shape[1]/mask.sum(axis=1)
            temp_loss.sum().backward()
            updater.step()
            metric.add(temp_loss.sum(),temp_loss.numel())
            if (i+1)%(num_batches//5)==0 or i==num_batches-1:
                print(f"epoch {epoch+(i+1)/num_batches:.3f} : loss={metric[0]/metric[1]:.3f}")
    print(f"loss={metric[0]/metric[1]:.3f},{metric[1]/timer.stop():.3f}tokens/sec on {str(device)}")

#寻找与输入单词语义最相似的单词
def get_similar_tokens(query_token,k,embed,vocab):
    W=embed.weight.data #(vocab_size,embedding_dim)
    x=W[vocab[query_token]]
    cos=torch.mv(W,x)/torch.sqrt(torch.sum(W*W,dim=1)*torch.sum(x*x)+1e-9)
    topk=torch.topk(cos,k=k+1)[1]   #只取索引，不取数值
    topk=topk.cpu().numpy().astype("int32")
    for i in topk[1:]:  #第一个相似度最高的肯定是自己，跳过
        print(f"cosine sim={float(cos[i]):.3f}:{vocab.to_tokens(i)}")

if __name__=="__main__":
    batch_size=512
    max_window_size=5
    num_noise_word=5
    data_iter,vocab=load_data_ptb(batch_size,max_window_size,num_noise_word)
    embed_size=100
    net=nn.Sequential(nn.Embedding(num_embeddings=len(vocab),embedding_dim=embed_size),nn.Embedding(num_embeddings=len(vocab),embedding_dim=embed_size))
    learning_rate=0.002
    num_epochs=5
    train(net,data_iter,learning_rate,num_epochs)
    get_similar_tokens("chip",3,net[0],vocab)