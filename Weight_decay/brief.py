import torch 
from torch import nn
from class_function_reused import synthetic_data
from class_function_reused import load_data
from class_function_reused import Accumulator

def evaluate_loss(net,data_iter,loss):
    metric=Accumulator(2)
    for x,y in data_iter:
        temp_outputs=net(x)
        y=y.reshape(temp_outputs.shape)
        temp_loss=loss(temp_outputs,y)
        metric.add(temp_loss.sum(),y.numel())
    return metric[0]/metric[1]

def train_brief(param_weight_decay):
    net=nn.Sequential(nn.Linear(num_inputs,1))
    for param in net.parameters():
        param.data.normal_()
    loss=nn.MSELoss(reduction="none")
    num_epochs=100
    learning_rate=0.003
    updater=torch.optim.SGD([{"params":net[0].weight,"weight_decay":param_weight_decay},{"params":net[0].bias}],lr=learning_rate)
    print(f"set param of weight_decay = {param_weight_decay}")
    for epoch in range(0,num_epochs):
        for x,y in train_iter:
            updater.zero_grad()
            temp_loss=loss(net(x),y)
            temp_loss.mean().backward()
            updater.step()
        if (epoch+1)%5==0:
            print(f"epoch{epoch+1}: train_loss={evaluate_loss(net,train_iter,loss)} , test_loss={evaluate_loss(net,test_iter,loss)}")
    print("L2 norm of weights:",net[0].weight.norm().item())

number_train,number_test=20,100
num_inputs=100
batch_size=5

true_weights=torch.ones((num_inputs,1))*0.01
true_bias=0.05
train_features,train_labels=synthetic_data(true_weights,true_bias,number_train)
test_features,test_labels=synthetic_data(true_weights,true_bias,number_test)
train_iter=load_data(train_features,train_labels,batch_size,is_train=True)
test_iter=load_data(test_features,test_labels,batch_size,is_train=False)

train_brief(0)
train_brief(20)

#find it seems that the brief one works less efficiently than the from_zero scheme , so let the param larger