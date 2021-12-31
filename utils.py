import torch
from torch.autograd import Variable
import torch.nn.functional as F
import warnings
import numpy as np
import torch.nn as nn
import random
import os
try:
    from autoattack import AutoAttack
except:
    print('no matter')

    
def inf_fgsm(model,
            origin,
            label,
            eps=0.03125,
            random_init = False,
            loss_fn = torch.nn.CrossEntropyLoss()
            ):
    device = origin.device
    if random_init:
        random_start = torch.FloatTensor(origin.size()).uniform_(-eps, eps).to(device)
        X_t = Variable(torch.clamp(origin+random_start,0,1),requires_grad=True)
    else:
        X_t = Variable(origin,requires_grad=True)
    logits = model(X_t)
    loss = loss_fn(logits,label)
    loss.backward()
    x_tmp = X_t+eps * torch.sign(X_t.grad)
    perturb = torch.clamp(x_tmp - origin,-eps,eps)
    X_t = Variable(torch.clamp(origin+perturb,0,1),requires_grad=True)
    return X_t.detach()
    

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
        
class LabelSmoothingLoss(nn.Module):
    #label smoothing(LS) loss
    def __init__(self, classes=10, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    
def inf_pgd(model,
            origin,
            label,
            iter_time,
            eps = 0.03125,
            step_size = 0.008,
            random_init = True,
            loss_fn = torch.nn.CrossEntropyLoss()
            ):
    loss_func = loss_fn
    device = origin.device
    if random_init:
        random_start = torch.FloatTensor(origin.size()).uniform_(-eps, eps).to(device)
        X_t = Variable(torch.clamp(origin+random_start,0,1),requires_grad=True)
        #X_t.requires_grad = True
    else:
        X_t = Variable(origin,requires_grad=True)
    for i in range(iter_time):
        logits = model(X_t)

        loss = loss_func(logits,label)
        loss.backward()
        x_tmp = X_t+step_size * torch.sign(X_t.grad)
        perturb = torch.clamp(x_tmp-origin,-eps,eps)
        X_t = Variable(torch.clamp(origin+perturb,0,1),requires_grad=True)
        #X_t.requires_grad = True
    return X_t.detach()

def inf_pgd_with_logits_scale(model,
            origin,
            label,
            iter_time,
            alpha = 1,
            eps = 0.03125,
            step_size = 0.008,
            random_init = True,
            loss_fn = torch.nn.CrossEntropyLoss()
            ):
    loss_func = loss_fn
    device = origin.device
    if random_init:
        random_start = torch.FloatTensor(origin.size()).uniform_(-eps, eps).to(device)
        X_t = Variable(torch.clamp(origin+random_start,0,1),requires_grad=True)
        #X_t.requires_grad = True
    else:
        X_t = Variable(origin,requires_grad=True)
    for i in range(iter_time):
        logits = model(X_t)

        loss = loss_func(alpha*logits,label)
        loss.backward()
        x_tmp = X_t+step_size * torch.sign(X_t.grad)
        perturb = torch.clamp(x_tmp-origin,-eps,eps)
        X_t = Variable(torch.clamp(origin+perturb,0,1),requires_grad=True)
        #X_t.requires_grad = True
    return X_t.detach()

def acc_test(model,image,label):
    batchsize = image.size()[0]
    logits = model(image)
    label = label.view(-1)
    max_pos = torch.argmax(logits,dim=1)
    corr = (label == max_pos).int().sum()
    err = batchsize - corr
    return err,corr
    
def cw_loss(logits,label):
    #loss function of c&w attack
    bk = Boolkiller('torch1.2', device=logits.device)
    logit_tmp = logits.detach().clone()
    label_one_hot = bk.get_one_hot(label, logits.size())
    logit_tmp[label_one_hot] = -10000000
    target_label = torch.argmax(logit_tmp,dim=1)
    target_one_hot = bk.get_one_hot(target_label, logits.size())

    other_max_logit = logits[target_one_hot]
    label_logit = logits[label_one_hot]
    return (other_max_logit-label_logit).mean()

class Boolkiller(object):
    def __init__(self,version,device):
        if version > 'torch1.1':
            self.handler = BoolHandlerTorch1_2(device)
        if version <= 'torch1.1':
            self.handler = BoolHandlerTorch1_1(device)

    def get_one_hot(self,label,size):
        return self.handler.get_one_hot(label,size)

    def bool_and(self,x,y):
        return self.handler.bool_and(x,y)

    def bool_or(self,x,y):
        return self.handler.bool_or(x,y)
    
    def bool_nor(self,x):
        return self.handler.bool_nor(x)
    

class BoolHandler(object):
    def __init__(self,device):
        self.device = device
    def get_one_hot(self):
        warnings.warn('abstract method,need rewrite')
    
    def bool_and(self):
        warnings.warn('abstract method,need rewrite')

    def bool_or(self):
        warnings.warn('abstract method,need rewrite')
    
    def bool_nor(self):
        warnings.warn('abstract method,need rewrite')

class BoolHandlerTorch1_1(BoolHandler):
    def __init__(self,device):
        super(BoolHandlerTorch1_1,self).__init__(device)
    def get_one_hot(self,label,size):
        label = label.view(-1)
        one_hot = torch.zeros(size).to(self.device)
        one_hot[torch.arange(size[0]), label] = 1
        return one_hot.type(torch.uint8)
    
    def bool_and(self,x,y):
        return (x+y) == 2

    def bool_or(self,x,y):
        return (x+y) == 1
    
    def bool_nor(self,x):
        return x==0

class BoolHandlerTorch1_2(BoolHandler):
    def __init__(self,device):
        super(BoolHandlerTorch1_2,self).__init__(device)
    def get_one_hot(self,label,size):
        one_hot= torch.zeros(size, dtype=torch.bool).to(self.device)
        label = label.reshape(-1,1)
        one_hot.scatter_(1, label, 1)
        return one_hot

    def bool_and(self,x,y):
        return x&y
    
    def bool_or(self,x,y):
        return x|y
    
    def bool_nor(self,x):
        return ~x
    
if __name__ == '__main__':
    pass



