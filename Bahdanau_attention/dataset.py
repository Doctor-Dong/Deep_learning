import torch
import os
import urllib.request
import zipfile
from class_function_reused import Vocab
from class_function_reused import load_array

def read_data_nmt(save_dir="data",zip_name="fra-eng.zip"):
    os.makedirs(save_dir,exist_ok=True)
    url="http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip"
    zip_path=os.path.join(save_dir,zip_name)
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url,zip_path)
    txt_path=os.path.join(save_dir,"fra-eng","fra.txt")
    if not os.path.exists(txt_path):
        with zipfile.ZipFile(zip_path,"r") as zf:
            zf.extractall(path=save_dir)
    with open(txt_path,"r",encoding="utf-8") as f:
        return f.read()
    
def preprocess_nmt(text):
    def no_space(char,pre_char):
        return char in set(",.!?") and pre_char!=" "
    text=text.replace("\u202f"," ").replace("\xa0"," ").lower()
    outputs=[" "+char if i>0 and no_space(char,text[i-1]) else char for i,char in enumerate(text)]
    return "".join(outputs)

def tokenize_nmt(text,num_examples=None):
    source=[]
    target=[]
    for i,line in enumerate(text.split("\n")):
        if num_examples and i>num_examples:
            break
        parts=line.split("\t")
        if len(parts)==2:
            source.append(parts[0].split(" "))
            target.append(parts[1].split(" "))
    return source,target

def truncate_pad(line,num_steps,padding_token):
    if len(line)>num_steps:
        return line[:num_steps]
    return line+[padding_token]*(num_steps-len(line))

def build_array_nmt(lines,vocab,num_steps):
    lines=[vocab[l] for l in lines]
    lines=[l+[vocab["<eos>"]] for l in lines]
    array=torch.tensor([truncate_pad(l,num_steps,vocab["<pad>"]) for l in lines])
    valid_len=(array!=vocab["<pad>"]).type(torch.int32).sum(1)
    return array,valid_len

def load_data_nmt(batch_size,num_steps,num_examples=600):
    text=preprocess_nmt(read_data_nmt())
    source,target=tokenize_nmt(text,num_examples)
    source_vocab=Vocab(source,min_freq=2,reserved_tokens=["<pad>","<bos>","<eos>"])
    target_vocab=Vocab(target,min_freq=2,reserved_tokens=["<pad>","<bos>","<eos>"])
    source_array,source_valid_len=build_array_nmt(source,source_vocab,num_steps)
    target_array,target_valid_len=build_array_nmt(target,target_vocab,num_steps)
    data_arrays=(source_array,source_valid_len,target_array,target_valid_len)
    data_iter=load_array(data_arrays,batch_size)
    return data_iter,source_vocab,target_vocab

if __name__=="__main__":
    raw_text=read_data_nmt()
    print(raw_text[:75])
    text=preprocess_nmt(raw_text)
    print(text[:80])
    source,target=tokenize_nmt(text)
    print(source[:6])
    print(target[:6])
    source_vocab=Vocab(source,min_freq=2,reserved_tokens=["<pad>","<bos>","<eos>"])
    print(len(source_vocab))
    print(truncate_pad(source_vocab[source[0]],10,source_vocab["<pad>"]))
    train_iter,source_vocab,target_vocab=load_data_nmt(batch_size=2,num_steps=8)
    for x,x_valid_len,y,y_valid_len in train_iter:
        print(x)
        print(x_valid_len)
        print(y)
        print(y_valid_len)
        break