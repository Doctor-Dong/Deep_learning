import os
import urllib.request
import zipfile
import torch
import random
from class_function_reused import get_tokens_and_segments
from class_function_reused import tokenize
from class_function_reused import Vocab
from class_function_reused import get_dataloader_workers

def download_wikitext2(root="data"):
    url="https://raw.githubusercontent.com/LogSSim/deeplearning_d2l_classes/main/class14_BERT/wikitext-2-v1.zip"
    data_dir=os.path.join(root,"wikitext-2")
    zip_path=os.path.join(root,"wikitext-2-v1.zip")
    os.makedirs(root,exist_ok=True)
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url,zip_path)
    if not os.path.exists(data_dir):
        with zipfile.ZipFile(zip_path,"r") as z:
            z.extractall(data_dir)
    return data_dir

def read_wikitext2(data_dir):
    file_name=os.path.join(data_dir,"wikitext-2","wiki.train.tokens")
    with open(file_name,"r",encoding="utf-8") as f:
        lines=f.readlines()
    paragraphs=[line.strip().lower().split(" . ") for line in lines if len(line.split(" . "))>=2]
    random.shuffle(paragraphs)
    return paragraphs

#辅助函数
#生成下一句预测任务的数据
def _get_next_sentence(sentence,next_sentence,paragraphs):
    if random.random()<0.5:
        is_next=True
    else:
        next_sentence=random.choice(random.choice(paragraphs))
        is_next=False
    return sentence,next_sentence,is_next

def _get_nsp_data_from_paragraph(paragraph,paragraphs,max_len):
    nsp_data_from_paragraph=[]
    for i in range(0,len(paragraph)-1):
        tokens_a,tokens_b,is_next=_get_next_sentence(paragraph[i],paragraph[i+1],paragraphs)
        if len(tokens_a)+len(tokens_b)+3>max_len:   #有一个"<cls>"词元和两个"<sep>"词元
            continue
        tokens,segments=get_tokens_and_segments(tokens_a,tokens_b)
        nsp_data_from_paragraph.append((tokens,segments,is_next))
    return nsp_data_from_paragraph

#生成遮蔽语言模型任务的数据
def _replace_mlm_tokens(tokens,candidate_pred_positions,num_mlm_preds,vocab):
    mlm_input_tokens=[token for token in tokens]
    pred_positions_and_labels=[]
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels)>=num_mlm_preds:
            break
        masked_token=None
        if random.random()<0.8: #80%替换为"<mask>"
            masked_token="<mask>"
        else:
            if random.random()<0.5: #10%保持不变
                masked_token=tokens[mlm_pred_position]
            else:   #10%替换为随机词
                masked_token=random.choice(vocab.index_to_token)
        mlm_input_tokens[mlm_pred_position]=masked_token
        pred_positions_and_labels.append((mlm_pred_position,tokens[mlm_pred_position]))
    return mlm_input_tokens,pred_positions_and_labels

def _get_mlm_data_from_tokens(tokens,vocab):
    candidate_pred_positions=[]
    for i,token in enumerate(tokens):
        if token in ["<cls>","<sep>"]:
            continue
        candidate_pred_positions.append(i)
    num_mlm_preds=max(1,round(len(tokens)*0.15))
    mlm_input_tokens,pred_positions_and_labels=_replace_mlm_tokens(tokens,candidate_pred_positions,num_mlm_preds,vocab)
    pred_positions_and_labels=sorted(pred_positions_and_labels,key=lambda x:x[0])
    pred_positions=[v[0] for v in pred_positions_and_labels]
    mlm_pred_labels=[v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens],pred_positions,vocab[mlm_pred_labels]

#将文本转换为预训练数据集
def _pad_bert_inputs(examples,max_len,vocab):
    max_num_mlm_preds=round(max_len*0.15)
    all_token_ids=[]
    all_segments=[]
    valid_lens=[]
    all_pred_positions=[]
    all_mlm_weights=[]
    all_mlm_labels=[]
    nsp_labels=[]
    for (token_ids,pred_positions,mlm_pred_label_ids,segments,is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids+[vocab["<pad>"]]*(max_len-len(token_ids)),dtype=torch.long))
        all_segments.append(torch.tensor(segments+[0]*(max_len-len(segments)),dtype=torch.long))
        valid_lens.append(torch.tensor(len(token_ids),dtype=torch.float32)) #不包括"<pad>"
        all_pred_positions.append(torch.tensor(pred_positions+[0]*(max_num_mlm_preds-len(pred_positions)),dtype=torch.long))
        all_mlm_weights.append(torch.tensor([1.0]*len(mlm_pred_label_ids)+[0.0]*(max_num_mlm_preds-len(pred_positions)),dtype=torch.float32))
        #填充词元的预测通过乘以权重0在损失中过滤掉
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids+[0]*(max_num_mlm_preds-len(mlm_pred_label_ids)),dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next,dtype=torch.long))
    return (all_token_ids,all_segments,valid_lens,all_pred_positions,all_mlm_weights,all_mlm_labels,nsp_labels)

class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self,paragraphs,max_len):
        paragraphs=[tokenize(paragraph,token="word") for paragraph in paragraphs]   #二维句子字符串列表->三维词元列表
        sentences=[sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab=Vocab(sentences,reserved_tokens=["<pad>","<mask>","<cls>","<sep>"],min_freq=5)
        #获取下一句子预测任务的数据
        examples=[]
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(paragraph,paragraphs,max_len))
        #获取遮蔽语言模型任务的数据
        examples=[(_get_mlm_data_from_tokens(tokens,self.vocab)+(segments,is_next)) for tokens,segments,is_next in examples]
        (self.all_token_ids,self.all_segments,self.valid_lens,self.all_pred_positions,self.all_mlm_weights,self.all_mlm_labels,self.nsp_labels)=_pad_bert_inputs(examples,max_len,self.vocab)

    def __getitem__(self,index):
        return (self.all_token_ids[index],self.all_segments[index],self.valid_lens[index],self.all_pred_positions[index],self.all_mlm_weights[index],self.all_mlm_labels[index],self.nsp_labels[index])
    
    def __len__(self):
        return len(self.all_token_ids)
    
def load_data_wiki(batch_size,max_len):
    num_workers=get_dataloader_workers()
    data_dir=download_wikitext2()
    paragraphs=read_wikitext2(data_dir)
    train_set=_WikiTextDataset(paragraphs,max_len)
    train_iter=torch.utils.data.DataLoader(train_set,batch_size,shuffle=True,num_workers=num_workers)
    return train_iter,train_set.vocab

if __name__=="__main__":
    batch_size=512
    max_len=64
    train_iter,vocab=load_data_wiki(batch_size,max_len)
    for (tokens_X,segments_X,valid_lens_x,pred_positions_X,mlm_weights_X,mlm_Y,nsp_y) in train_iter:
        print(tokens_X.shape,segments_X.shape,valid_lens_x.shape,pred_positions_X.shape,mlm_weights_X.shape,mlm_Y.shape,nsp_y.shape)
        break
    print(len(vocab))