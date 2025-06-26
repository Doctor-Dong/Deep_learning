from collections import Counter
from torch.utils import data

def count_freq(tokens):
    if len(tokens)==0 or isinstance(tokens[0],list):
        tokens=[token for line in tokens for token in line]
    return Counter(tokens)

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
    
def load_array(data_arrays,batch_size,is_train=True):
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)