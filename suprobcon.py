import torch
import torch.nn as nn
import torch.distributed as dist
import diffdist
from utils import inf_pgd
from models.resnet import *
from supcon import *
import torch.nn.functional as F
from torch.autograd import Variable

class SupRobConLoss(nn.Module):
    def __init__(self,temperature=0.07,world_size=1,lamda=0.5):
        super(SupRobConLoss, self).__init__()
        
        self.temperature = temperature
        self.world_size = world_size
        self.lamda = lamda
    
    def get_mask(self,labels):
        device = labels.device
        
        bs = len(labels)
        N = 2 * bs 
        label_double = torch.cat((labels,labels),0)
        mask1 = torch.ones((N, N)).to(device)
        mask1 = mask1.fill_diagonal_(0)
        
        label_horizontal = label_double.clone()
        label_vertical = label_horizontal.view(-1,1).to(device)
        
        mask2 = label_horizontal-label_vertical
        mask2 = mask2 == 0
        mask2 = mask2.float()
        
        return mask1,mask2
        
    def forward(self,z0,z1,z1_adv,labels):
        bs = z0.size()[0]
        N = bs*self.world_size
        
        if self.world_size > 1:
            z0_list = [torch.zeros_like(z0) for _ in range(self.world_size)]
            z1_list = [torch.zeros_like(z1) for _ in range(self.world_size)]
            z1_adv_list = [torch.zeros_like(z1_adv) for _ in range(self.world_size)]
            label_lis = [torch.zeros_like(labels) for _ in range(self.world_size)]
            
            z0_list = diffdist.functional.all_gather(z0_list, z0)
            z1_list = diffdist.functional.all_gather(z1_list, z1)
            z1_adv_list = diffdist.functional.all_gather(z1_adv_list, z1_adv)
            label_lis = diffdist.functional.all_gather(label_lis, labels)
            
            z0 = torch.cat(z0_list,dim=0)
            z1 = torch.cat(z1_list,dim=0)
            z1_adv = torch.cat(z1_adv_list,dim=0)
            labels = torch.cat(label_lis,dim=0)

        mask_eye, mask_same_class = self.get_mask(labels)
        mask_final = mask_eye * mask_same_class
        
        vector_0_1 = torch.cat((z0,z1),dim=0)
        vector_0_1_adv = torch.cat((z0,z1_adv),dim=0)
        
        sim_mat_0_1 = nn.CosineSimilarity(dim=2)(vector_0_1.unsqueeze(1), vector_0_1.unsqueeze(0)) / self.temperature
        sim_mat_0_1_adv = nn.CosineSimilarity(dim=2)(vector_0_1_adv.unsqueeze(1), vector_0_1_adv.unsqueeze(0)) / self.temperature
        
        sim_mat = self.lamda * sim_mat_0_1 + (1-self.lamda) * sim_mat_0_1_adv
        
        exp_logits = torch.exp(sim_mat) * mask_eye
        
        log_prob = sim_mat - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask_final * log_prob).sum(1) / mask_final.sum(1)
        
        loss = - mean_log_prob_pos.view(2,N).mean()

        return loss
        
        
class SupRobConModel(nn.Module):
    def __init__(self,backbone,feature_d = 512,mlp_d = 128):
        super(SupRobConModel, self).__init__()
        self.backbone = backbone
        
        self.mlp_head = nn.Sequential(
            nn.Linear(feature_d, feature_d, bias=False),
            nn.ReLU(),
            nn.Linear(feature_d, mlp_d, bias=False),
        )
        self.cur_epoch = 0
    
    def init_suprobcon(self, temperature, world_size,beta=1):
        self.supcon = SupConLoss(temperature=temperature,world_size=world_size)
        self.beta = beta
        
    def init_pgd(self,eps,step_size,iter_time):
        self.eps = eps
        self.step_size = step_size
        self.iter_time = iter_time
        
    def init_simclr(self,temperature,world_size):
        self.simclr = SupConLoss(temperature=temperature,world_size=world_size)
        
    
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
    
    def forward(self,x0,x1=None,label=None):
        if label == None and x1 == None:
            out = self.get_logits(x0)
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
                adv0,adv1 = self.inf_pgd_cl(x0,x1,label=label,iter_time=self.iter_time,eps=self.eps,step_size=self.step_size)
                self.train()
                #dist.barrier()
                bs = x0.size()[0]
                x_combine = torch.cat((adv0,adv1),dim=0)
                
                features = self.get_feature(x_combine)
                #feature_0 = features[0:bs]
                #feature_1 = features[bs:2*bs]
                feature_adv_0 = features[0:bs]
                feature_adv_1 = features[bs:]
                
                
                #z0 = self.mlp_head(feature_0).unsqueeze(1)
                #z1 = self.mlp_head(feature_1).unsqueeze(1)
                z0_adv = self.mlp_head(feature_adv_0).unsqueeze(1)
                z1_adv = self.mlp_head(feature_adv_1).unsqueeze(1)
                
                #z_left = torch.cat((z0,z1,z0_adv),dim=1)
                #z_right =torch.cat((z0,z1,z1_adv),dim=1)
                z_rob = torch.cat((z0_adv,z1_adv),dim=1)
                
                #loss_suprobcon = self.suprobcon(z0,z1,z1_adv,label)
                loss_suprobcon = self.supcon(z_rob,label)
                
                #loss_rob = self.simclr(z_rob)
                
                #loss_suprobcon = loss_supcon + self.beta * loss_rob
                
                feature_1_adv_bk = feature_adv_1.clone().detach()
                logits = self.backbone.linear(feature_1_adv_bk)
                loss_ce_detach = torch.nn.CrossEntropyLoss()(logits,label)
                
                loss = loss_suprobcon + loss_ce_detach
                
                return loss

    def inf_pgd_cl(self,x1,x2,label=None,eps=8/255,step_size=2/255,iter_time=10,random_init=True):
        device = x1.device
        if random_init:
            random_start_1 = torch.FloatTensor(x1.size()).uniform_(-eps, eps).to(device)
            X_1 = Variable(torch.clamp(x1+random_start_1,0,1),requires_grad=True)
            
            random_start_2 = torch.FloatTensor(x2.size()).uniform_(-eps, eps).to(device)
            X_2 = Variable(torch.clamp(x2+random_start_2,0,1),requires_grad=True)
            
        else:
            X_1 = Variable(x1,requires_grad=True)
            X_2 = Variable(x2,requires_grad=True)
            
        for i in range(iter_time):
            feature_X1 = self.get_feature(X_1).unsqueeze(dim=1)
            feature_X2 = self.get_feature(X_2).unsqueeze(dim=1)
            
            feature = torch.cat((feature_X1,feature_X2),dim=1)
            loss = self.simclr(feature,labels=label)
            loss.backward()
            x1_tmp = X_1+step_size * torch.sign(X_1.grad)
            perturb1 = torch.clamp(x1_tmp-x1,-eps,eps)
            X_1 = Variable(torch.clamp(x1+perturb1,0,1),requires_grad=True)
            x2_tmp = X_2+step_size * torch.sign(X_2.grad)
            perturb2 = torch.clamp(x2_tmp-x2,-eps,eps)
            X_2 = Variable(torch.clamp(x2+perturb2,0,1),requires_grad=True)
        return X_1.detach(),X_2.detach()
    


    
        
        
        
        