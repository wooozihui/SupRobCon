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



    
class Resnet18_feature(nn.Module):
    def __init__(self,resnet18,class_num=10):
        super(Resnet18_feature, self).__init__()
        self.resnet = resnet18
        self.class_num = class_num
        #self.last_bn = nn.BatchNorm2d(512)
    def get_feature(self,x):
        out = F.relu(self.resnet.bn1(self.resnet.conv1(x)))
        out = self.resnet.layer1(out)
        out = self.resnet.layer2(out)
        out = self.resnet.layer3(out)
        out = self.resnet.layer4(out)
        out = F.avg_pool2d(out, 4)
        #out = self.last_bn(out)
        #print(out.size())
        out = out.view(out.size(0), -1)
        return out
    
    def fc_predict(self,x):
        out = self.resnet.linear(x)
        return out
    
    def forward(self,x):
        out = self.get_feature(x)
        out = self.fc_predict(out)
        return out    
    
def inf_pgd_with_prior(model,
            origin,
            prior,
            label,
            iter_time,
            eps = 0.03125,
            step_size = 0.008,
            random_init = True,
            loss_fn = torch.nn.CrossEntropyLoss()
            ):
    loss_func = loss_fn
    device = origin.device
    prior = prior.to(device)
    X_t = Variable(torch.clamp(origin+prior,0,1),requires_grad=True)
    for i in range(iter_time):
        logits = model(X_t)

        loss = loss_func(logits,label)
        loss.backward()
        x_tmp = X_t+step_size * torch.sign(X_t.grad)
        perturb = torch.clamp(x_tmp-origin,-eps,eps)
        X_t = Variable(torch.clamp(origin+perturb,0,1),requires_grad=True)
        #X_t.requires_grad = True
    return X_t.detach()    

#### Prior-based-FGSM-attack ########
### prior = adversarial - origin ###
def inf_fgsm_with_prior(model,
            origin,
            prior,
            label,
            eps=0.03125,
            step_size = 0.008,
            loss_fn = torch.nn.CrossEntropyLoss()
            ):
    device = origin.device
    prior = prior.to(device)
    X_t = Variable(torch.clamp(origin+prior,0,1),requires_grad=True)
    logits = model(X_t)
    loss = loss_fn(logits,label)
    loss.backward()
    x_tmp = X_t+step_size * torch.sign(X_t.grad)
    perturb = torch.clamp(x_tmp - origin,-eps,eps)
    X_t = Variable(torch.clamp(origin+perturb,0,1),requires_grad=True)
    return X_t.detach()    
    
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
    
    
def anti_loss(logits,labels):
    celoss = torch.nn.CrossEntropyLoss()(logits,labels)
    loss_entropy = - torch.softmax(logits,dim=1).detach() * F.log_softmax(logits, dim=1)
    loss_entropy = loss_entropy.sum(dim=1).mean()
    return -loss_entropy - celoss      
    
def autoattack_test_by_class(model,testloader,class_i = 0, n_ex=1000,batch_size=100,eps=0.03125,norm='Linf',version='standard',device='cuda'):
    print("test class: "+str(class_i))
    l_x = []
    l_y = []
    for i, (x, y) in enumerate(testloader):
        pos = y==class_i
        if pos.sum() >0:
            x = x[pos]
            y = y[pos]
            l_x.append(x)
            l_y.append(y)
    x_test = torch.cat(l_x, 0)
    y_test = torch.cat(l_y, 0)
    print(x_test.size())
    print(y_test.size())

    if version == "standard":
        print("use standard version of auto attack, this may be very time consuming. I recommend using rand version of auto attack first")

    adversary = AutoAttack(model, norm=norm, eps=eps, version=version,device=device)
    #adversary = AutoAttack(model, norm=norm, eps=eps, attacks_to_run=[],version='sb')
    
    with torch.no_grad():
        adv = adversary.run_standard_evaluation(x_test[:n_ex], y_test[:n_ex],bs=batch_size)    
    
def pgd_test_by_class(model,testloader,device,classnum,step_size = 0.008,pgd_time = 20,loss_fn=torch.nn.CrossEntropyLoss()):
	nature_right = 0
	total = 0
	adv_right = 0
	for i,(image,label) in enumerate(testloader):
		image = image.to(device)
		label = label.to(device)
		#if i == 10:
		#	break
		pos = label == classnum
		if pos.sum()>0:
			label = label[pos]
			image = image[pos]

			batchsize = image.size()[0]
			err0,right0 = acc_test(model,image,label)
			adv_image = inf_pgd(model,image,label,eps=0.03125,iter_time=pgd_time,step_size = step_size,loss_fn=loss_fn)
			err1,right1 = acc_test(model,adv_image,label)
			nature_right+=right0
			adv_right+=right1
			total+=batchsize
	total = float(total)
	nature_acc = float(nature_right)/total
	adv_acc = float(adv_right)/total
	return nature_acc,adv_acc    
    
def set_bin_optimizer(model, init_lr=0.1,bin_lr=10,momentum=0.9,weight_decay=2e-4):
    params = [{'params': [p for p in model.parameters() if not getattr(p, 'bin_gate', False)]},
              {'params': [p for p in model.parameters() if getattr(p, 'bin_gate', False)], 
               'lr': init_lr * bin_lr, 'weight_decay': 0}]
    optimizer = torch.optim.SGD(params, 
            lr=init_lr, 
            momentum=momentum,
            weight_decay=weight_decay)
    return optimizer

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



def acc_test(model,image,label):
    batchsize = image.size()[0]
    logits = model(image)
    label = label.view(-1)
    max_pos = torch.argmax(logits,dim=1)
    corr = (label == max_pos).int().sum()
    err = batchsize - corr
    return err,corr


def white_box_test(model,testloader,device,test_pic_num = 1000,eps=0.03125,step_size = 0.008,pgd_time = 20,slogan=None,loss_fn=torch.nn.CrossEntropyLoss()):
    '''
    White-box attack robustness test.
        :loss_fn: torch.nn.CrossEntropyLoss() --> PGD attack
                 cw_loss --> C&W attack
        :return: natural accuracy & robustness
    '''
    natural_corr = 0
    total = 0
    adv_corr = 0
    #print("acc testing.......")
    for i,(image,label) in enumerate(testloader):
        image = image.to(device)
        label = label.to(device)
        batchsize = image.size()[0]
        err0,corr0 = acc_test(model,image,label)
        adv_image = inf_pgd(model,image,label,eps=eps,iter_time=pgd_time,step_size = step_size,loss_fn=loss_fn)
        err1,corr1 = acc_test(model,adv_image,label)
        natural_corr+=corr0
        adv_corr+=corr1
        total+=batchsize
        tested = (i+1)*batchsize
        #print(tested)
        if tested >= test_pic_num:
            break
    total = float(total)
    natural_acc = float(natural_corr)/total
    adv_acc = float(adv_corr)/total
    if slogan!=None:
        print(slogan,"acc:",natural_acc,"rob:",adv_acc)
    else:
        print("acc:",natural_acc,"rob:",adv_acc)
    return natural_acc,adv_acc

def autoattack_test(model,testloader,n_ex=1000,batch_size=100,eps=0.03125,norm='Linf',version='standard',device='cuda'):
    '''
    AutoAttack robustness test.
        :param model: torch model returns the logits and takes input with components in [0, 1] (NCHW format expected),
        :testloader: testloader,
        :n_ex: total number of images to be attacked,
        :batch_size: you know what batch_size means,
        :param eps: eps is the bound on the norm of the adversarial perturbations,
        :param norm: norm = ['Linf' | 'L2'] is the norm of the threat model,
        :param version:version = ['standard' | 'rand'] 'standard' uses the standard version of AA. 
        :return: attack accuracy
    '''
    # load testloader and prepare dataset to attack
    l_x = []
    l_y = []
    for i, (x, y) in enumerate(testloader):
        l_x.append(x)
        l_y.append(y)
    x_test = torch.cat(l_x, 0)
    y_test = torch.cat(l_y, 0)

    if version == "standard":
        print("use standard version of auto attack, this may be very time consuming. I recommend using rand version of auto attack first")

    adversary = AutoAttack(model, norm=norm, eps=eps, version=version,device=device)
    #adversary = AutoAttack(model, norm=norm, eps=eps, attacks_to_run=[],version='sb')
    
    with torch.no_grad():
        adv = adversary.run_standard_evaluation(x_test[:n_ex], y_test[:n_ex],bs=batch_size)
    # print("robust accuracy: {:.2%}'.format(robust_accuracy)")
    
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

def adjust_learning_rate(optimizer, epoch,bin = False,init_lr=0.1,schedule='trades_fixed',end=100,bin_para=10):
    """decrease the learning rate"""
    lr = init_lr
    # schedule from TRADES repo (different from paper due to bug there)
    if schedule == 'trades':
        if epoch >= 0.75 * end:
            lr = init_lr * 0.1
    # schedule as in TRADES paper
    elif schedule == 'trades_fixed':
        if epoch >= 0.75 * end:
            lr = init_lr * 0.1
        if epoch >= 0.9 * end:
            lr = init_lr * 0.01
        if epoch >= end:
            lr = init_lr * 0.001
    # cosine schedule
    elif schedule == 'cosine':
        lr = init_lr * 0.5 * (1 + np.cos((epoch - 1) / end * np.pi))
    # schedule as in WRN paper
    elif schedule == 'wrn':
        if epoch >= 0.3 * end:
            lr = init_lr * 0.2
        if epoch >= 0.6 * end:
            lr = init_lr * 0.2 * 0.2
        if epoch >= 0.8 * end:
            lr = init_lr * 0.2 * 0.2 * 0.2
    else:
        raise ValueError('Unkown LR schedule %s' % schedule)
    if bin:
        # adjust learning rate for the bin norm layer
        print('adjust bin')
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr*bin_para
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

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



