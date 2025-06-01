import torch
from data_load import load_fashion_mnist_data
from data_load import get_fashion_mnist_labels
from class_function_reused import Accumulator
from class_function_reused import sgd

#define softmax
def softmax(X):
    X_exp=torch.exp(X)
    sum_row=X_exp.sum(dim=1,keepdim=True)
    return X_exp/sum_row

#define model
def net(X):
    return softmax(torch.matmul(X.reshape((-1,weights.shape[0])),weights)+bias)

#define loss
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(0,len(y_hat)),y])

#define the number of predicting rightly
def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=y_hat.argmax(axis=1)
    cmp=(y_hat.type(y.dtype)==y)
    return float(cmp.type(y.dtype).sum())

#compute the ratio
def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric=Accumulator(2)
    with torch.no_grad():
        for x,y in data_iter:
            metric.add(accuracy(net(x),y),y.numel())
    return metric[0]/metric[1]

#define training per epoch
def train_epoch(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric=Accumulator(3)
    for x,y in train_iter:
        y_hat=net(x)
        temp_loss=loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            temp_loss.mean().backward()
            updater.step()
        else:
            temp_loss.sum().backward()
            updater(x.shape[0])
        metric.add(float(temp_loss.sum()),accuracy(y_hat,y),y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]

#define training
def train(net,train_iter,test_iter,loss,updater,num_epochs):
    for epoch in range(0,num_epochs):
        train_loss,train_acc=train_epoch(net,train_iter,loss,updater)
        test_acc=evaluate_accuracy(net,test_iter)
        print(f"epoch{epoch+1}: train_loss={train_loss:.5f} , train_acc={train_acc:.5f} , test_acc={test_acc:.5f}")
    assert train_loss<0.5,"train_loss is too high"
    assert train_acc<=1 and train_acc>0.7,"train_acc is too low"
    assert test_acc<=1 and test_acc>0.7,"test_acc is too low"

#define updater
def updater(batch_size):
    return sgd([weights,bias],learning_rate,batch_size)

#define prediction
def predict(net,test_iter,number=20):
    for x,y in test_iter:
        break
    trues=get_fashion_mnist_labels(y)
    preds=get_fashion_mnist_labels(net(x).argmax(axis=1))
    print(f"true    labels:{trues[0:number]}")
    print(f"predict labels:{preds[0:number]}")

batch_size=256
train_iter,test_iter=load_fashion_mnist_data(batch_size)

num_inputs=784  #28*28
num_outputs=10
weights=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
bias=torch.zeros(num_outputs,requires_grad=True)

learning_rate=0.1
num_epochs=10

train(net,train_iter,test_iter,cross_entropy,updater,num_epochs)
predict(net,test_iter)