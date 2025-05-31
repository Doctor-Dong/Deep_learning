import torch
from torch.utils import data
from torch import nn

def synthetic_data(weight,bias,num_examples):
    x=torch.normal(0,1,(num_examples,len(weight)))
    y=torch.matmul(x,weight)+bias
    y+=torch.normal(0,0.01,y.shape) #随机噪声
    return x,y.reshape((-1,1))

#read dataset
def load_data(features,labels,batch_size,is_train=True):
    dataset=data.TensorDataset(features,labels)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

true_weights=torch.tensor([2,-3.4])
true_bias=4.2
features,labels=synthetic_data(true_weights,true_bias,1000)

batch_size=10
data_iter=load_data(features,labels,batch_size)

#define model
net=nn.Sequential(nn.Linear(2,1))

#init params
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

#define loss
loss=nn.MSELoss()

#define trainier
trainer=torch.optim.SGD(net.parameters(),lr=0.03)

#training
num_epochs=3
for epoch in range(0,num_epochs):
    for x,y in data_iter:
        loss_temp=loss(net(x),y)
        trainer.zero_grad()
        loss_temp.backward()
        trainer.step()
    with torch.no_grad():
        training_loss=loss(net(features),labels)
        print(f"epoch {epoch+1} : loss = {training_loss:f}")

print(f"true weights:{true_weights}")
print(f"computed weights:{net[0].weight.data}")
print(f"true bias:{true_bias}")
print(f"computed bias:{net[0].bias.data}")
