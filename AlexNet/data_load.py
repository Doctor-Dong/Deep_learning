import torchvision
from torchvision import transforms
from torch.utils import data

def get_dataloader_workers():
    return 0

def load_fashion_mnist_data(batch_size,resize=None):
    trans=[transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)
    mnist_train=torchvision.datasets.FashionMNIST(root="data",train=True,transform=trans,download=True)
    mnist_test=torchvision.datasets.FashionMNIST(root="data",train=False,transform=trans,download=True)
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers()),data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=get_dataloader_workers()))

def get_fashion_mnist_labels(labels):
    text_lables=["t-shirt","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle boot"]
    return [text_lables[int(index)] for index in labels]