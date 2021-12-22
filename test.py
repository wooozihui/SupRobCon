import torch
import torch.nn as nn
import torch.distributed as dist
import diffdist
from utils import inf_pgd
from models.resnet import *
from supcon import *
import torch.nn.functional as F

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
        print(mask_same_class)
        print(mask_final)
        
        
        vector_0_1 = torch.cat((z0,z1),dim=0)
        vector_0_1_adv = torch.cat((z0,z1_adv),dim=0)
        
        sim_mat_0_1 = nn.CosineSimilarity(dim=2)(vector_0_1.unsqueeze(1), vector_0_1.unsqueeze(0)) / self.temperature
        sim_mat_0_1_adv = nn.CosineSimilarity(dim=2)(vector_0_1_adv.unsqueeze(1), vector_0_1_adv.unsqueeze(0)) / self.temperature
        
        print(sim_mat_0_1.size())
        
        sim_mat = self.lamda * sim_mat_0_1 + (1-self.lamda) * sim_mat_0_1_adv
        
        exp_logits = torch.exp(sim_mat) * mask_eye
        print(exp_logits)
        
        log_prob = sim_mat - torch.log(exp_logits.sum(1, keepdim=True))
        
        print(log_prob)
        
        mean_log_prob_pos = (mask_final * log_prob).sum(1) / mask_final.sum(1)
        
        print(mean_log_prob_pos.size())
        
        loss = - mean_log_prob_pos.view(2,N).mean()
        print(loss)
        return loss
    
if __name__ == '__main__':
    a = torch.ones(4,128)/10
    label = torch.Tensor([1,2,3,1]).long()
    cri = SupRobConLoss()
    cri(a,a,a,label)