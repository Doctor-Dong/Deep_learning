import torch
from torch import nn
from dataset import load_data_wiki
from bert import BERTModel
from GPU import try_all_gpus
from class_function_reused import Timer
from class_function_reused import Accumulator
from class_function_reused import get_tokens_and_segments

def _get_batch_loss_bert(net,loss,vocab_size,tokens_X,segments_X,valid_lens_x,pred_positions_X,mlm_weights_X,mlm_Y,nsp_y):
    _,mlm_Y_hat,nsp_Y_hat=net(tokens_X,segments_X,valid_lens_x.reshape(-1),pred_positions_X)
    #计算遮蔽语言模型损失
    mlm_loss=loss(mlm_Y_hat.reshape(-1,vocab_size),mlm_Y.reshape(-1)).reshape(-1,1)
    mlm_loss=mlm_loss*mlm_weights_X.reshape(-1,1)
    mlm_loss=mlm_loss.sum()/(mlm_weights_X.sum()+1e-8)
    #计算下一句子预测任务的损失
    nsp_loss=loss(nsp_Y_hat,nsp_y).mean()
    temp_loss=mlm_loss+nsp_loss
    return mlm_loss,nsp_loss,temp_loss

def train_bert(train_iter,net,loss,vocab_size,devices,num_steps):
    net=nn.DataParallel(net,device_ids=devices).to(devices[0])
    trainer=torch.optim.Adam(net.parameters(),lr=0.01)
    step=0
    num_steps_reached=False
    metric=Accumulator(4)
    timer=Timer()
    while step<num_steps and not num_steps_reached:
        for tokens_X,segments_X,valid_lens_x,pred_positions_X,mlm_weights_X,mlm_Y,nsp_y in train_iter:
            tokens_X=tokens_X.to(devices[0])
            segments_X=segments_X.to(devices[0])
            valid_lens_x=valid_lens_x.to(devices[0])
            pred_positions_X=pred_positions_X.to(devices[0])
            mlm_weights_X=mlm_weights_X.to(devices[0])
            mlm_Y=mlm_Y.to(devices[0])
            nsp_y=nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_loss,nsp_loss,temp_loss=_get_batch_loss_bert(net,loss,vocab_size,tokens_X,segments_X,valid_lens_x,pred_positions_X,mlm_weights_X,mlm_Y,nsp_y)
            temp_loss.backward()
            trainer.step()
            metric.add(mlm_loss,nsp_loss,tokens_X.shape[0],1)
            timer.stop()
            print(f"step{step+1}:mlm_loss={metric[0]/metric[3]:.3f},nsp_loss={metric[1]/metric[3]:.3f}")
            step+=1
            if step==num_steps:
                num_steps_reached=True
                break
    print(f"total MLM loss={metric[0]/metric[3]:.3f}")
    print(f"total NSP loss={metric[1]/metric[3]:.3f}")
    print(f"{metric[2]/timer.total_time():.3f} sentence pairs/sec on {str(devices)}")

#用BERT表示文本
def get_bert_encoding(net,devices,tokens_a,tokens_b=None):
    tokens,segments=get_tokens_and_segments(tokens_a,tokens_b)
    token_ids=torch.tensor(vocab[tokens],device=devices[0]).unsqueeze(0)
    segments=torch.tensor(segments,device=devices[0]).unsqueeze(0)
    valid_len=torch.tensor(len(tokens),device=devices[0]).unsqueeze(0)
    encoded_X,_,_=net(token_ids,segments,valid_len)
    return encoded_X

if __name__=="__main__":
    batch_size=512
    max_len=64
    train_iter,vocab=load_data_wiki(batch_size,max_len)
    net=BERTModel(len(vocab),num_hiddens=128,normalized_shape=[128],ffn_num_inputs=128,ffn_num_hiddens=256,num_heads=2,num_layers=2,dropout=0.2,key_size=128,query_size=128,value_size=128,hid_in_features=128,mlm_in_features=128,nsp_in_features=128)
    devices=try_all_gpus()
    loss=nn.CrossEntropyLoss(reduction="none")
    train_bert(train_iter,net,loss,len(vocab),devices,num_steps=50)
    
    tokens_a=["a","crane","is","flying"]
    encoded_text=get_bert_encoding(net,devices,tokens_a)
    encoded_text_cls=encoded_text[:,0,:]
    encoded_text_crane=encoded_text[:,2,:]
    print(encoded_text.shape)
    print(encoded_text_cls.shape)
    print(encoded_text_crane[0][:3])

    tokens_a=["a","crane","driver","came"]
    tokens_b=["he","just","left"]
    encoded_pair=get_bert_encoding(net,devices,tokens_a,tokens_b)
    encoded_pair_cls=encoded_pair[:,0,:]
    encoded_pair_crane=encoded_pair[:,2,:]
    print(encoded_pair.shape)
    print(encoded_pair_cls.shape)
    print(encoded_pair_crane[0][:3])