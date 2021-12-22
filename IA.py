import torch
import torch.nn as nn
import torch.distributed as dist
import diffdist
from utils import inf_pgd
from models.resnet import *
from simclr import *

### IAL: the identity align loss ###
class IAL(nn.Module):
    def __init__(self,batch_size, temperature, world_size):
        super(IAL,self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
    
    def get_mask(self,labels):
        device = labels.device
        label_horizontal = labels.clone().to(device)
        label_vertical = label_horizontal.view(-1,1).to(device)
        
        N = self.batch_size*self.world_size
        mask1 = torch.ones((N, N), dtype=bool).to(device)
        mask1 = mask1.fill_diagonal_(0)
        
        mask2 = label_horizontal-label_vertical
        mask2 = mask2 == 0
        
        mask = mask1 & mask2
        mask = mask.float()
        return mask

    def forward(self,z,z_adv,labels):
        N = self.batch_size*self.world_size
        
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
        sim_adv = nn.CosineSimilarity(dim=1)(z,z_adv)/ self.temperature
        sim_adv_e = torch.exp(sim_adv)
        
        sim = nn.CosineSimilarity(dim=2)(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_e = torch.exp(sim)
        
        mask = self.get_mask(labels)
        mask = mask.to(sim_e.device)
        
        intra_class_sim = sim_e * mask
        intra_class_sim_sum = intra_class_sim.sum(dim=1).view(-1)
        
        loss = -torch.log(sim_adv_e/(sim_adv_e+intra_class_sim_sum))
        loss = loss.sum()
        loss = loss/N
        return loss

class IA_model(nn.Module):
    def __init__(self,backbone,feature_d = 512,mlp_d = 128):
        super(IA_model, self).__init__()
        self.backbone = backbone
        
        self.mlp_head = nn.Sequential(
            nn.Linear(feature_d, feature_d, bias=False),
            nn.ReLU(),
            nn.Linear(feature_d, mlp_d, bias=False),
        )
    
    def init_IAL(self,batch_size, temperature, world_size,beta=1):
        self.IAL = IAL(batch_size, temperature, world_size)
        self.beta = beta


    def init_pgd(self,eps,step_size,iter_time):
        self.eps = eps
        self.step_size = step_size
        self.iter_time = iter_time
    
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
            self.eval()
            advs = inf_pgd(self,x,label,iter_time=self.iter_time,eps=self.eps,step_size=self.step_size)
            self.train()
            #dist.barrier()
            
            feature = self.get_feature(x)
            feature_adv = self.get_feature(advs)
            
            adv_logits = self.backbone.linear(feature_adv)
            
            #z = self.mlp_head(feature)
            #z_adv = self.mlp_head(feature_adv)
            
            loss_ce = torch.nn.CrossEntropyLoss()(adv_logits,label)
            #loss_ia = self.IAL(z,z_adv,label)
            loss_ia = self.IAL(feature,feature_adv,label)

            loss = loss_ce + self.beta*loss_ia
            return loss,loss_ce,loss_ia
            
class SupCon_wrapper(IA_model):
    
    pass
            

if __name__ == "__main__":
    model = ResNet18().cuda(device="cuda:1")
    ia_model = IA_model(model).cuda(device="cuda:1")
    ia_model.init_IAL(10,0.1,1)
    
    z = torch.randn(10,3,32,32).cuda(device="cuda:1")
    label = torch.Tensor([0,1,3,1,3,4,5,6,7,9]).long().cuda(device="cuda:1")


    loss,loss_ce,loss_ia = ia_model(z,label)
    print(loss)
    print(loss_ce)
    print(loss_ia)
    loss.backward()

    
        
        
        
        