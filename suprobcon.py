import torch
import torch.nn as nn
import torch.distributed as dist
import diffdist
from utils import inf_pgd,inf_pgd_with_logits_scale
from models.resnet import *
from supcon import *
import torch.nn.functional as F
from torch.autograd import Variable

class SupRobConLoss(nn.Module):
    def __init__(self,temperature=0.07,world_size=1):
        super(SupRobConLoss, self).__init__()
        
        self.temperature = temperature
        self.world_size = world_size
    
    def get_mask(self,labels):
        device = labels.device
        
        N = len(labels)

        mask1 = torch.ones((N, N)).to(device)
        mask1 = mask1.fill_diagonal_(0)
        
        label_horizontal = labels.clone()
        label_vertical = label_horizontal.view(-1,1).to(device)
        
        mask2 = label_horizontal-label_vertical
        mask2 = mask2 == 0
        mask3 = ~mask2
        mask2 = mask2.float()
        mask3 = mask3.float()
        return mask1,mask2,mask3
        
    def forward(self,z,z_adv,labels):
        if self.world_size > 1:
            z_list = [torch.zeros_like(z) for _ in range(self.world_size)]
            z_adv_list = [torch.zeros_like(z_adv) for _ in range(self.world_size)]
            label_lis = [torch.zeros_like(labels) for _ in range(self.world_size)]
            
            z_list = diffdist.functional.all_gather(z_list, z)
            z_adv_list = diffdist.functional.all_gather(z_adv_list, z_adv)
            label_lis = diffdist.functional.all_gather(label_lis, labels)
            
            z = torch.cat(z_list,dim=0)
            z_adv = torch.cat(z_adv_list,dim=0)
            labels = torch.cat(label_lis,dim=0)

        mask_eye, mask_same_class, mask_diff_class = self.get_mask(labels)
        
        sim_mat = nn.CosineSimilarity(dim=2)(z_adv.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        up = torch.exp(sim_mat)
        exp_other_class = up * mask_diff_class
        
        down = up + exp_other_class.sum(1, keepdim=True)
        
        loss = -(torch.log(up/down) * mask_same_class).sum(1) / mask_same_class.sum(1)
        loss = loss.mean()

        return loss
        
        
class SupRobConModel(nn.Module):
    def __init__(self,backbone,feature_d = 512,mlp_d = 128):
        super(SupRobConModel, self).__init__()
        self.backbone = backbone
        
        self.mlp_head = nn.Sequential(
            nn.Linear(feature_d, feature_d),
            nn.BatchNorm1d(feature_d),
            nn.ReLU(inplace=True),
            nn.Linear(feature_d, mlp_d),
            #nn.ReLU(),
        )
        self.cur_epoch = 0
    
    def init_suprobcon(self, temperature, world_size):
        self.suprobcon = SupRobConLoss(temperature=temperature,world_size=world_size)
        
    def init_pgd(self,eps,step_size,iter_time):
        self.eps = eps
        self.step_size = step_size
        self.iter_time = iter_time
    
    '''
    not use
    def update_epoch(self,epoch):
        self.cur_epoch=epoch
        
    def warmup_beta(self):
        cur_epoch = self.cur_epoch
        cur_beta = self.beta
        if self.beta_warmup_epoch !=None :
            if cur_epoch < self.beta_warmup_epoch:
                cur_beta = self.beta*((cur_epoch+1)/self.beta_warmup_epoch)
        return cur_beta
    '''
    
    def get_feature(self,x):
        out = F.relu(self.backbone.bn1(self.backbone.conv1(x)))
        out = self.backbone.layer1(out)
        out = self.backbone.layer2(out)
        out = self.backbone.layer3(out)
        out = self.backbone.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
    
    def get_logits(self,x):
        out = self.get_feature(x)
        out = self.backbone.linear(out)
        return out
    
    def forward(self,x,label=None):
        if label == None:
            out = self.get_logits(x)
            return out
        else:
            if label == None:
                # simclr loss #
                pass
            else:
                ## get beta ##
                # not use beta = self.warmup_beta()
                ### end #####
                self.eval()
                advs = inf_pgd(self,x,label,iter_time=self.iter_time,eps=self.eps,step_size=self.step_size)
                self.train()
                
                features = self.get_feature(x)
                features_adv = self.get_feature(advs)
                
                z = self.mlp_head(features)
                z_adv = self.mlp_head(features_adv)
                
                loss_suprobcon = self.suprobcon(z,z_adv,label)
                
                feature_adv_bk = features_adv.clone().detach()
                logits = self.backbone.linear(feature_adv_bk)
                loss_ce_detach = torch.nn.CrossEntropyLoss()(logits,label)
                
                loss = loss_suprobcon + loss_ce_detach
                
                return loss,loss_suprobcon,loss_ce_detach
            
    def inf_pgd_src(self,x,label,eps=8/255,step_size=2/255,iter_time=10,random_init=True):     
        device = x.device
        if random_init:
            random_start = torch.FloatTensor(x.size()).uniform_(-eps, eps).to(device)
            X_t = Variable(torch.clamp(x+random_start,0,1),requires_grad=True)
        else:
            X_t = Variable(x,requires_grad=True)
        for i in range(iter_time):
            features = self.get_feature(x)
            features_adv = self.get_feature(X_t)
                
            z = self.mlp_head(features)
            z_adv = self.mlp_head(features_adv)
            
            loss = self.suprobcon(z_adv,z,label)
            loss.backward()
            x_tmp = X_t+step_size * torch.sign(X_t.grad)
            perturb = torch.clamp(x_tmp-x,-eps,eps)
            X_t = Variable(torch.clamp(x+perturb,0,1),requires_grad=True)
        return X_t.detach()

    
    


    
        
        
        
        