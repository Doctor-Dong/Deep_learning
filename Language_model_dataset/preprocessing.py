import os
import re
from urllib import request
from collections import Counter

#download the text
def download_text(url:str,save_path:str):
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    if not os.path.exists(save_path):
        request.urlretrieve(url,save_path)

def read_time_machine(data_dir="data"):
    url="http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"
    fname=os.path.join(data_dir,"timemachine.txt")
    download_text(url,fname)
    with open(fname,"r") as f:
        lines=f.readlines()
    return [re.sub("[^A-Za-z]+"," ",line).strip().lower() for line in lines]

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

#define vocab
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
    
#summary
def load_corpus_time_machine(max_tokens=-1):
    lines=read_time_machine()
    tokens=tokenize(lines,token="char") #这里使用char模式
    vocab=Vocab(tokens)
    corpus=[vocab[token] for line in tokens for token in line]
    if max_tokens>0:
        corpus=corpus[:max_tokens]
    return corpus,vocab

if __name__=="__main__":
    corpus,vocab=load_corpus_time_machine()
    print(list(vocab.token_to_index.items())[0:10])
    print(len(corpus))
    print(len(vocab))