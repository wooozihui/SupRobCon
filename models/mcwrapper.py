import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

import numpy as np
import random
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter

_feature_nums = 30

class my_MaxPool2d(Module):


    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(my_MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        input = input.transpose(3,1)


        input = F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
        input = input.transpose(3,1).contiguous()

        return input

    def __repr__(self):
        kh, kw = _pair(self.kernel_size)
        dh, dw = _pair(self.stride)
        padh, padw = _pair(self.padding)
        dilh, dilw = _pair(self.dilation)
        padding_str = ', padding=(' + str(padh) + ', ' + str(padw) + ')' \
            if padh != 0 or padw != 0 else ''
        dilation_str = (', dilation=(' + str(dilh) + ', ' + str(dilw) + ')'
                        if dilh != 0 and dilw != 0 else '')
        ceil_str = ', ceil_mode=' + str(self.ceil_mode)
        return self.__class__.__name__ + '(' \
            + 'kernel_size=(' + str(kh) + ', ' + str(kw) + ')' \
            + ', stride=(' + str(dh) + ', ' + str(dw) + ')' \
            + padding_str + dilation_str + ceil_str + ')'


class my_AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(my_AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        input = input.transpose(3,1)
        input = F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)
        input = input.transpose(3,1).contiguous()

        return input


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', ceil_mode=' + str(self.ceil_mode) \
            + ', count_include_pad=' + str(self.count_include_pad) + ')'

class w_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(w_BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class w_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(w_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class w_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(w_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        #self.in_planes = 64
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, _feature_nums, num_blocks[3], stride=2)
        self.linear = nn.Linear(_feature_nums*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
  

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #print(out.size())
        out = self.layer1(out)
        #print(out.size())
        out = self.layer2(out)
        #print(out.size())
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def w_ResNet18():
    return w_ResNet(w_BasicBlock, [2, 2, 2, 2])


def Mask(nb_batch,feature_channels,class_num=10,device='cuda'):
    channels = int(feature_channels/class_num)
    tmp = int(round(channels/2))
    tmp2 = int(channels - tmp)
    foo = [1] * tmp + [0] *  tmp2
    bar = []
    for i in range(class_num):
        random.shuffle(foo)
        bar += foo
    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch,class_num*channels,1,1)
    bar = torch.from_numpy(bar)
    bar = bar.to(device)
    bar = Variable(bar)
    return bar




def supervisor(x,targets,height,cnum):
        mask = Mask(x.size(0),_feature_nums,10,x.device)
        branch = x
        branch = branch.reshape(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))
        branch = F.softmax(branch,2)
        branch = branch.reshape(branch.size(0),branch.size(1), x.size(2), x.size(2))
        branch = my_MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch)  
        branch = branch.reshape(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))
        loss_2 = 1.0 - 1.0*torch.mean(torch.sum(branch,2))/cnum # set margin = 3.0

        branch_1 = x * mask 

        branch_1 = my_MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch_1)  
        branch_1 = nn.AvgPool2d(kernel_size=(height,height))(branch_1)
        branch_1 = branch_1.view(branch_1.size(0), -1)

        loss_1 = torch.nn.CrossEntropyLoss()(branch_1, targets)
        
        return [loss_1, loss_2] 

class MCwrapperResNet18(nn.Module):
    def __init__(self):
        super(MCwrapperResNet18, self).__init__()
        self.model = w_ResNet18()
        '''
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(_feature_nums),
            #nn.Dropout(0.5),
            nn.Linear(_feature_nums, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(512, 10),
        )
        '''
        self.classifier = self.model.linear
        
    def get_feature(self,x):
        out = F.relu(self.model.bn1(self.model.conv1(x)))
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        return out
    
    def get_logits(self,x):
        out = F.max_pool2d(x, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
        
    def train_forward(self,x,labels):
        feature = self.get_feature(x)
        MC_loss = supervisor(feature,labels,height=4,cnum=3)
        logits = self.get_logits(feature)
        return logits,MC_loss
    
    def forward(self, x):
        feature = self.get_feature(x)
        logits = self.get_logits(feature)
        return logits



if __name__ == '__main__':
    aaa = MCwrapperResNet18().eval()
    x = torch.Tensor(1,3,32,32)
    features = aaa(x)
    print(features.size())
    label = torch.Tensor([0]).long()
    features,mcloss = aaa.train_forward(x,label)
    print(features.size())
    print(mcloss)