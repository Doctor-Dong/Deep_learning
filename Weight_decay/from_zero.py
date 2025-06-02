import torch
from class_function_reused import synthetic_data
from class_function_reused import load_data
from class_function_reused import linreg
from class_function_reused import squared_loss
from class_function_reused import sgd
from class_function_reused import Accumulator

def init_params():
    weights=torch.normal(0,1,size=(num_inputs,1),requires_grad=True)
    bias=torch.zeros(1,requires_grad=True)
    return [weights,bias]

def L2_penalty(weights):
    return torch.sum(weights.pow(2))/2

def evaluate_loss(net,data_iter,loss):
    metric=Accumulator(2)
    for x,y in data_iter:
        temp_outputs=net(x)
        y=y.reshape(temp_outputs.shape)
        temp_loss=loss(temp_outputs,y)
        metric.add(temp_loss.sum(),y.numel())
    return metric[0]/metric[1]

def train(lambd):
    weights,bias=init_params()
    net=lambda X:linreg(X,weights,bias)
    loss=squared_loss
    num_epochs=100
    learning_rate=0.003
    print(f"set lambd = {lambd}")
    for epoch in range(0,num_epochs):
        for x,y in train_iter:
            temp_loss=loss(net(x),y)+lambd*L2_penalty(weights)
            temp_loss.sum().backward()
            sgd([weights,bias],learning_rate,batch_size)
        if (epoch+1)%5==0:
            print(f"epoch{epoch+1}: train_loss={evaluate_loss(net,train_iter,loss)} , test_loss={evaluate_loss(net,test_iter,loss)}")
    print("L2 norm of weights:",torch.norm(weights).item())

number_train,number_test=20,100
num_inputs=100
batch_size=5

true_weights=torch.ones((num_inputs,1))*0.01
true_bias=0.05
train_features,train_labels=synthetic_data(true_weights,true_bias,number_train)
test_features,test_labels=synthetic_data(true_weights,true_bias,number_test)
train_iter=load_data(train_features,train_labels,batch_size,is_train=True)
test_iter=load_data(test_features,test_labels,batch_size,is_train=False)

train(lambd=0)
train(lambd=3)