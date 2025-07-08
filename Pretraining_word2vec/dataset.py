import os
import urllib.request
import zipfile
import random
import math
import torch
import torch.utils.data.dataloader
from class_function_reused import Vocab
from class_function_reused import count_freq
from class_function_reused import get_dataloader_workers

def download_extract(data_dir="data",url="http://d2l-data.s3-accelerate.amazonaws.com/ptb.zip"):
    os.makedirs(data_dir,exist_ok=True)
    zip_path=os.path.join(data_dir,"ptb.zip")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url,zip_path)
    extract_dir=os.path.join(data_dir,"ptb")
    if not os.path.isdir(extract_dir):
        with zipfile.ZipFile(zip_path,"r") as z:
            z.extractall(data_dir)
    return extract_dir

def read_ptb(data_dir="data"):
    data_dir=download_extract(data_dir)
    with open(os.path.join(data_dir,"ptb.train.txt")) as f:
        raw_text=f.read()
    return [line.split() for line in raw_text.split("\n")]

#下采样
def subsample(sentences,vocab):
    sentences=[[token for token in line if vocab[token]!=vocab.unk] for line in sentences]
    counter=count_freq(sentences)
    num_tokens=sum(counter.values())
    
    def keep(token):
        return random.uniform(0,1)<math.sqrt(1e-4/counter[token]*num_tokens)
    
    return ([[token for token in line if keep(token)] for line in sentences],counter)

#中心词和上下文词的提取
def get_centers_and_contexts(corpus,max_window_size):
    centers=[]
    contexts=[]
    for line in corpus:
        if len(line)<2:
            continue
        centers+=line
        for i in range(0,len(line)):
            window_size=random.randint(1,max_window_size)
            indices=list(range(max(0,i-window_size),min(len(line),i+window_size+1)))
            indices.remove(i)
            contexts.append([line[index] for index in indices])
    return centers,contexts

#负采样
class RandomGenerator:
    def __init__(self,sampling_weights):
        self.population=list(range(1,len(sampling_weights)+1))
        self.sampling_weights=sampling_weights
        self.candidates=[]
        self.i=0

    def draw(self):
        if self.i==len(self.candidates):
            self.candidates=random.choices(self.population,self.sampling_weights,k=10000)
            self.i=0
        self.i+=1
        return self.candidates[self.i-1]

def get_negatives(all_contexts,vocab,counter,K):
    sampling_weights=[counter[vocab.to_tokens(i)]**0.75 for i in range(1,len(vocab))]
    #索引从1开始，因为索引0是词表中排除的未知标记
    all_negatives=[]
    generator=RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives=[]
        while len(negatives)<len(contexts)*K:   #对一对中心词和上下文词，随机抽取K个噪声词
            neg=generator.draw()
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

#小批量加载训练实例
def batchify(data): #data=(center,context,negative)
    max_len=max(len(c)+len(n) for _,c,n in data)
    centers=[]
    contexts_negatives=[]
    masks=[]    #用来标注是否是填充
    labels=[]   #用来标注是否是上下文词正例
    for center,context,negative in data:
        temp_len=len(context)+len(negative)
        centers+=[center]
        contexts_negatives+=[context+negative+[0]*(max_len-temp_len)]
        masks+=[[1]*temp_len+[0]*(max_len-temp_len)]
        labels+=[[1]*len(context)+[0]*(max_len-len(context))]
    return (torch.tensor(centers).reshape((-1,1)),torch.tensor(contexts_negatives),torch.tensor(masks),torch.tensor(labels))

#整合
def load_data_ptb(batch_size,max_window_size,num_noise_words):
    num_workers=get_dataloader_workers()
    sentences=read_ptb()
    vocab=Vocab(sentences,min_freq=10)
    subsampled,counter=subsample(sentences,vocab)
    corpus=[vocab[line] for line in subsampled]
    all_centers,all_contexts=get_centers_and_contexts(corpus,max_window_size)
    all_negatives=get_negatives(all_contexts,vocab,counter,num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self,centers,contexts,negatives):
            assert len(centers)==len(contexts)==len(negatives)
            self.centers=centers
            self.contexts=contexts
            self.negatives=negatives
        
        def __getitem__(self,index):
            return (self.centers[index],self.contexts[index],self.negatives[index])
        
        def __len__(self):
            return len(self.centers)
    
    dataset=PTBDataset(all_centers,all_contexts,all_negatives)
    data_iter=torch.utils.data.DataLoader(dataset,batch_size,shuffle=True,collate_fn=batchify,num_workers=num_workers)
    return data_iter,vocab

if __name__=="__main__":
    sentences=read_ptb()
    print(len(sentences))
    vocab=Vocab(sentences,min_freq=10)
    print(len(vocab))
    subsampled,counter=subsample(sentences,vocab)

    def compare_counts(token):
        print(f"{token}的数量：之前={sum([l.count(token) for l in sentences])}，之后={sum([l.count(token) for l in subsampled])}")

    compare_counts("the")
    compare_counts("join")

    corpus=[vocab[line] for line in subsampled]
    print(corpus[:3])
    all_centers,all_contexts=get_centers_and_contexts(corpus,max_window_size=5)
    print(f"中心词-上下文词对的数量={sum([len(contexts) for contexts in all_contexts])}")
    all_negatives=get_negatives(all_contexts,vocab,counter,K=5)
    print(all_negatives[:3])

    data_iter,vocab=load_data_ptb(512,5,5)
    names=["centers","contexts_negatives","masks","labels"]
    for batch in data_iter:
        for name,data in zip(names,batch):
            print(name," shape:",data.shape)
        break