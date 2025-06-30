import torch
from torch import nn

if __name__=="__main__":
    n_train=50
    x_train,_=torch.sort(torch.rand(n_train)*5)

    def f(x):
        return 2*torch.sin(x)+x**0.8
    
    y_train=f(x_train)+torch.normal(0.0,0.5,(n_train,))
    x_test=torch.arange(0,5,0.1)
    y_truth=f(x_test)
    n_test=len(x_test)
    print(x_train)
    print(y_train)

    #平均汇聚
    y_hat=torch.repeat_interleave(y_train.mean(),n_test)
    print(y_hat)

    #非参数注意力汇聚(Nadaraya-Watson)
    X_repeat=x_test.repeat_interleave(n_train).reshape((-1,n_train))
    attention_weights=nn.functional.softmax(-(X_repeat-x_train)**2/2,dim=1)
    y_hat=torch.matmul(attention_weights,y_train)
    print(y_hat)

    #带参数注意力汇聚
    class NMKernelRegression(nn.Module):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            self.w=nn.Parameter(torch.rand((1,),requires_grad=True))
        
        def forward(self,queries,keys,values):
            queries=queries.repeat_interleave(keys.shape[1]).reshape((-1,keys.shape[1]))
            self.attention_weights=nn.functional.softmax(-((queries-keys)*self.w)**2/2,dim=1)
            return torch.bmm(self.attention_weights.unsqueeze(1),values.unsqueeze(-1)).reshape(-1)
    
    #train
    X_tile=x_train.repeat((n_train,1))
    Y_tile=y_train.repeat((n_train,1))
    keys=X_tile[(1-torch.eye(n_train)).type(torch.bool)].reshape((n_train,-1))  #(n_train,n_train-1)
    values=Y_tile[(1-torch.eye(n_train)).type(torch.bool)].reshape((n_train,-1))
    net=NMKernelRegression()
    loss=nn.MSELoss(reduction="none")
    updater=torch.optim.SGD(net.parameters(),lr=0.5)
    for epoch in range(0,5):
        updater.zero_grad()
        temp_loss=loss(net(x_train,keys,values),y_train)
        temp_loss.sum().backward()
        updater.step()
        print(f"epoch{epoch+1},loss={float(temp_loss.sum()):.6f}")
    
    keys=x_train.repeat((n_test,1))
    values=y_train.repeat((n_test,1))
    y_hat=net(x_test,keys,values)
    print(y_hat)
    print(y_truth)