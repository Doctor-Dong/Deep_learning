import torch
from torch import nn
from class_function_reused import load_data
from class_function_reused import evaluate_loss

#generate data
T=1000
time=torch.arange(1,T+1,dtype=torch.float32)
x=torch.sin(0.01*time)+torch.normal(0,0.2,(T,))

#generate feature-label pair
tau=4
features=torch.zeros((T-tau,tau))
for i in range(0,tau):
    features[:,i]=x[i:T-tau+i]
labels=x[tau:].reshape((-1,1))

batch_size=16
number_of_train=600
train_iter=load_data(features[:number_of_train],labels[:number_of_train],batch_size,is_train=True)

#define the net
def init_weights(module):
    if type(module)==nn.Linear:
        nn.init.xavier_uniform_(module.weight)

def get_net():
    net=nn.Sequential(nn.Linear(4,10),nn.ReLU(),nn.Linear(10,1))
    net.apply(init_weights)
    return net

#define train
def train(net,train_iter,loss,num_epochs,learning_rate):
    updater=torch.optim.Adam(net.parameters(),lr=learning_rate)
    for epoch in range(0,num_epochs):
        for X,y in train_iter:
            updater.zero_grad()
            temp_loss=loss(net(X),y)
            temp_loss.mean().backward()
            updater.step()
        print(f"epoch{epoch+1} : loss={evaluate_loss(net,train_iter,loss):f}")

net=get_net()
loss=nn.MSELoss(reduction="none")

train(net,train_iter,loss,num_epochs=5,learning_rate=0.01)

#define predict
#one-step
one_step_preds=net(features)
print("one-step prediction:")
for i in range(0,700):
    print(f"label={labels[i]} , predict={one_step_preds[i]}")

#multi-step
multi_step_preds=torch.zeros(T)
multi_step_preds[0:number_of_train+tau]=x[0:number_of_train+tau]
for i in range(number_of_train+tau,T):
    multi_step_preds[i]=net(multi_step_preds[i-tau:i].reshape((1,-1)))
print("multi-step prediction:")
for i in range(number_of_train+tau,T):
    print(f"label={x[i]} , predict={multi_step_preds[i]}")

max_step=64
step_features=torch.zeros((T-tau-max_step+1,tau+max_step))
for i in range(0,tau):
    step_features[:,i]=x[i:i+T-tau-max_step+1]
for i in range(tau,tau+max_step):
    step_features[:,i]=net(step_features[:,i-tau:i]).reshape(-1)
print(f"max_step={max_step}")
for i in range(0,T-tau-max_step+1):
    print(f"label={x[i+tau+max_step-1]} , predict={step_features[i][-1]}")