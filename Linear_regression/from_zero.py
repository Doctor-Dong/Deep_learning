import torch
import random

def synthetic_data(weight,bias,num_examples):
    x=torch.normal(0,1,(num_examples,len(weight)))
    y=torch.matmul(x,weight)+bias
    y+=torch.normal(0,0.01,y.shape) #随机噪声
    return x,y.reshape((-1,1))

def data_iter(batch_size,features,labels):
    num_examples=len(features)
    index=list(range(0,num_examples))
    random.shuffle(index)
    for i in range(0,num_examples,batch_size):
        batch_index=torch.tensor(index[i:min(i+batch_size,num_examples)])
        yield features[batch_index],labels[batch_index]

#define model
def linreg(x,weights,bias):
    return torch.matmul(x,weights)+bias

#define loss
def squared_loss(y_hat,y):
    return ((y_hat-y.reshape(y_hat.shape))**2)/2

#define sgd
def sgd(params,learning_rate,batch_size):
    with torch.no_grad():
        for param in params:
            param-=learning_rate*param.grad/batch_size
            param.grad.zero_()

#generate data
true_weight=torch.tensor([2,-3.4])
true_bias=4.2
features,labels=synthetic_data(true_weight,true_bias,1000)

batch_size=10

#init params
weights=torch.normal(0,0.01,size=(2,1),requires_grad=True)
bias=torch.zeros(1,requires_grad=True)

learning_rate=0.03
num_epochs=3
net=linreg
loss=squared_loss

for epoch in range(0,num_epochs):
    for x,y in data_iter(batch_size,features,labels):
        temp_loss=loss(net(x,weights,bias),y)
        temp_loss.sum().backward()
        sgd([weights,bias],learning_rate,batch_size)
    with torch.no_grad():
        training_loss=loss(net(features,weights,bias),labels)
        print(f"epoch : {epoch+1} , loss = {float(training_loss.mean()):f}")

print(f"true weights:{true_weight}")
print(f"computed weights:{weights}")
print(f"true bias:{true_bias}")
print(f"computed bias:{bias}")