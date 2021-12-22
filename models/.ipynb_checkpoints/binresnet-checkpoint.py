'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
#from .cosface_loss import CosineMarginProduct
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,normlayer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = normlayer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = normlayer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                normlayer(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,normlayer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = normlayer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = normlayer(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = normlayer(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                normlayer(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BINResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,norm_type=None):
        super(BINResNet, self).__init__()
        if norm_type == 'bn':
            from torch.nn import BatchNorm2d as Normlayer
        elif norm_type == 'in':
            from torch.nn import InstanceNorm2d as Normlayer
        elif norm_type == 'bin':
            from .batchinstancenorm import BatchInstanceNorm2d as Normlayer
        self.normlayer = functools.partial(Normlayer, affine=True)
        
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = self.normlayer(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,self.normlayer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
  

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    

class BINResNet_2d_featrue(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,norm_type=None):
        super(BINResNet_2d_featrue, self).__init__()
        if norm_type == 'bn':
            from torch.nn import BatchNorm2d as Normlayer
        elif norm_type == 'in':
            from torch.nn import InstanceNorm2d as Normlayer
        elif norm_type == 'bin':
            from .batchinstancenorm import BatchInstanceNorm2d as Normlayer
        self.normlayer = functools.partial(Normlayer, affine=True)
        
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = self.normlayer(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear1 = nn.Linear(512, 2)
        self.linear2 = nn.Linear(2, num_classes,bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,self.normlayer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
  

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out
    
    def get_feature(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out
    
    
class Featurer(nn.Module):
    def __init__(self,model):
        super(Featurer, self).__init__()
        self.model = model
    
    def forward(self,x):
        return self.model(x)
    
    def get_feature(self,x):
        out = F.relu(self.model.bn1(self.model.conv1(x)))
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
    def f2l(self,features):
        logits = self.model.linear(features)
        return logits

def BINResNet18(norm_type='bn',num_classes=10):
        return BINResNet(BasicBlock, [2, 2, 2, 2],norm_type=norm_type,num_classes=num_classes)

def BINResNet18_2d(norm_type='bn',num_classes=10):
        return BINResNet_2d_featrue(BasicBlock, [2, 2, 2, 2],norm_type=norm_type,num_classes=num_classes)
    
    
def BINResNet34(norm_type='bn'):
    return BINResNet(BasicBlock, [3, 4, 6, 3],norm_type=norm_type)


def BINResNet50(norm_type='bn'):
    return BINResNet(Bottleneck, [3, 4, 6, 3],norm_type=norm_type)


def BINResNet101(norm_type='bn'):
    return BINResNet(Bottleneck, [3, 4, 23, 3],norm_type=norm_type)


def ResNet152(norm_type='bn'):
    return ResNet(Bottleneck, [3, 8, 36, 3],norm_type=norm_type)

if __name__ == '__main__':
    #cosface = CosfaceResNet(BasicBlock, [2, 2, 2, 2],norm_type='bn',num_classes=10)
    model = BINResNet18_2d()
    a = torch.rand(10,3,32,32)
    label = torch.LongTensor([1]*10)
    out = model(a)
    feature = model.get_feature(a)
    print(out.size())
    print(feature.size())