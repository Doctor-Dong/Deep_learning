import torch
import random
from preprocessing import tokenize
from preprocessing import read_time_machine
from preprocessing import Vocab
from preprocessing import load_corpus_time_machine

#随机采样
def seq_data_iter_random(corpus,batch_size,num_steps):
    corpus=corpus[random.randint(0,num_steps-1):]   #随机偏移量
    num_subseqs=(len(corpus)-1)//num_steps  #因为label是下一个位置的词元，所以要-1
    initial_indices=list(range(0,num_subseqs*num_steps,num_steps))
    random.shuffle(initial_indices)

    def data(index):
        return corpus[index:index+num_steps]  
     
    num_batches=num_subseqs//batch_size
    for i in range(0,num_batches*batch_size,batch_size):
        initial_indices_per_batch=initial_indices[i:i+batch_size]
        X=[data(j) for j in initial_indices_per_batch]
        Y=[data(j+1) for j in initial_indices_per_batch]
        yield torch.tensor(X),torch.tensor(Y)

#顺序分区
def seq_data_iter_sequential(corpus,batch_size,num_steps):
    offset=random.randint(0,num_steps-1)
    num_tokens=((len(corpus)-offset-1)//batch_size)*batch_size
    X=torch.tensor(corpus[offset:offset+num_tokens])
    Y=torch.tensor(corpus[offset+1:offset+1+num_tokens])
    X=X.reshape((batch_size,-1))
    Y=Y.reshape((batch_size,-1))
    num_batches=X.shape[1]//num_steps
    for i in range(0,num_steps*num_batches,num_steps):
        X_temp=X[:,i:i+num_steps]
        Y_temp=Y[:,i:i+num_steps]
        yield X_temp,Y_temp

class SeqDataLoader:
    def __init__(self,batch_size,num_steps,use_random_iter,max_tokens):
        if use_random_iter:
            self.data_iter_fn=seq_data_iter_random
        else:
            self.data_iter_fn=seq_data_iter_sequential
        self.corpus,self.vocab=load_corpus_time_machine(max_tokens)
        self.batch_size=batch_size
        self.num_steps=num_steps
    
    def __iter__(self):
        return self.data_iter_fn(self.corpus,self.batch_size,self.num_steps)
    
def load_data_time_machine(batch_size,num_steps,use_random_iter=False,max_tokens=10000):
    data_iter=SeqDataLoader(batch_size,num_steps,use_random_iter,max_tokens)
    return data_iter,data_iter.vocab

if __name__=="__main__":
    tokens=tokenize(read_time_machine())
    corpus=[token for line in tokens for token in line]
    vocab=Vocab(tokens=corpus)
    print(vocab.token_freq[:10])
    #停用词

    #二元语法
    bigram_tokens=[pair for pair in zip(corpus[:-1],corpus[1:])]
    bigram_vocab=Vocab(tokens=bigram_tokens)
    print("bigram_token_freq:")
    print(bigram_vocab.token_freq[:10])

    #三元语法
    trigram_tokens=[triple for triple in zip(corpus[:-2],corpus[1:-1],corpus[2:])]
    trigram_vocab=Vocab(trigram_tokens)
    print("trigram_token_freq:")
    print(trigram_vocab.token_freq[:10])

    my_seq=list(range(0,35))
    print("random data_iter:")
    for x,y in seq_data_iter_random(my_seq,batch_size=2,num_steps=5):
        print("x:",x)
        print("y:",y)

    print("sequential data_iter:")
    for x,y in seq_data_iter_sequential(my_seq,batch_size=2,num_steps=5):
        print("x:",x)
        print("y:",y)