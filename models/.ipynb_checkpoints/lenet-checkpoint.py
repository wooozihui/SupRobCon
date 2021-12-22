'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):#a LeNet model
    def __init__(self):
        super(LeNet,self).__init__()
        #input[N,1,28,28]
        self.conv1 = nn.Conv2d(1,6,1)#[N,6,28,28]
        self.avepool = nn.AvgPool2d(2,2)#1.[N,6,14,14];2.[N,16,5,5]
        self.conv2 = nn.Conv2d(6,16,5)#[N,16,10,10]
        #flatten[N,400]
        self.fc1 = nn.Linear(400,100)#[N,100]
        self.fc2 = nn.Linear(100,84)#[N,84]
        self.fc3 = nn.Linear(84,10)#[N,10]
        return

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.avepool(x)
        x = F.relu(self.conv2(x))
        x = self.avepool(x)
        x = x.view(-1,400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x

if __name__ == '__main__':
    model = LeNet()
    image = torch.randn(10,1,28,28)
    logits = model(image)
    print(logits.size())