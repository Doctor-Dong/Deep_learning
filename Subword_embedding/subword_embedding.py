import collections

def get_max_freq_pair(token_freqs):
    pairs=collections.defaultdict(int)
    for token,freq in token_freqs.items():
        symbols=token.split()
        for i in range(0,len(symbols)-1):
            pairs[(symbols[i],symbols[i+1])]+=freq
    return max(pairs,key=pairs.get) #返回具有最大值的pairs键

def merge_symbols(max_freq_pair,token_freqs,symbols):   #合并最频繁的连续符号产生新符号
    symbols.append("".join(max_freq_pair))
    new_token_freqs=dict()
    for token,freq in token_freqs.items():
        new_token=token.replace(" ".join(max_freq_pair),"".join(max_freq_pair))
        new_token_freqs[new_token]=freq
    return new_token_freqs

def segment_BPE(tokens,symbols):    #贪心策略
    outputs=[]
    for token in tokens:
        start=0
        end=len(token)
        cur_output=[]
        while start<len(token) and start<end:
            if token[start:end] in symbols:
                cur_output.append(token[start:end])
                start=end
                end=len(token)
            else:
                end-=1
        if start<len(token):
            cur_output.append("[UNK]")
        outputs.append(" ".join(cur_output))
    return outputs

if __name__=="__main__":
    symbols=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","_","[UNK]"]
    raw_token_freqs={"fast_":4,"faster_":3,"tall_":5,"taller_":4}
    token_freqs={}
    for token,freq in raw_token_freqs.items():
        token_freqs[" ".join(list(token))]=freq
    print(raw_token_freqs)
    print(token_freqs)
    num_merges=10
    for i in range(0,num_merges):
        max_freq_pair=get_max_freq_pair(token_freqs)
        token_freqs=merge_symbols(max_freq_pair,token_freqs,symbols)
        print(f"第{i+1}次合并：",max_freq_pair)
    print(symbols)
    print(list(token_freqs.keys()))
    tokens=["tallest_","fatter_"]
    print(segment_BPE(tokens,symbols))