import torch
from torch import nn
from class_function_reused import Accumulator
from class_function_reused import accuracy
from class_function_reused import Timer

def evaluate_accuracy_gpu(net,data_iter,device=None):
    if isinstance(net,nn.Module):
        net.eval()
        if not device:
            device=next(iter(net.parameters())).device
    metric=Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            if isinstance(X,list):
                X=[x.to(device) for x in X]
            else:
                X=X.to(device)
            y=y.to(device)
            metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

def train(net,train_iter,test_iter,num_epochs,learning_rate,device):
    def init_weights(module):
        if type(module)==nn.Linear or type(module)==nn.Conv2d:
            nn.init.xavier_uniform_(module.weight)
    net.apply(init_weights)
    print("training on ",device)
    net.to(device)
    optimizer=torch.optim.SGD(net.parameters(),lr=learning_rate)
    loss=nn.CrossEntropyLoss()
    timer=Timer()
    num_batches=len(train_iter)
    for epoch in range(0,num_epochs):
        metric=Accumulator(3)
        net.train()
        for index,(X,y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X=X.to(device)
            y=y.to(device)
            y_hat=net(X)
            temp_loss=loss(y_hat,y)
            temp_loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(temp_loss*X.shape[0],accuracy(y_hat,y),X.shape[0])
            timer.stop()
            train_loss=metric[0]/metric[2]
            train_acc=metric[1]/metric[2]
            if (index+1)%(num_batches//5)==0 or index==num_batches-1:
                print(f"train_loss = {train_loss:.3f} , train_accuracy = {train_acc:.3f}")
        test_acc=evaluate_accuracy_gpu(net,test_iter)
        print(f"epoch{epoch+1} : test_accuracy={test_acc:.3f}")
    print(f"the whole training : train_loss={train_loss:.3f} , train_accuracy={train_acc:.3f} , test_accuracy={test_acc:.3f}")
    print(f"{metric[2]*num_epochs/timer.total_time():.1f} examples/second on {str(device)}")