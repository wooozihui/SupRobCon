import os
import torch
from autoattack import AutoAttack
from ddp_util import *
from models.resnet import *
from torchvision import transforms
import torchvision
import torch.distributed as dist
from utils import inf_pgd
from functools import partial
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Robustness test')
parser.add_argument('--local_rank',type=int, default=0)
parser.add_argument('--test_bs',type=int, default=250)
parser.add_argument("--model_path",type=str,default=None)
parser.add_argument("--load_best",type=bool,default=True)
parser.add_argument('--total_num',type=int, default=1000)
parser.add_argument('--attack_type',type=str,default="pgd") # option: AA, cw

parser.add_argument('--aa_version',type=str,default="base") 


parser.add_argument('--eps',type=float,default=8/255)
parser.add_argument('--step_size',type=float,default=2/255)
parser.add_argument('--iter_time',type=int,default=20)


args = parser.parse_args()
init_ddp(args)


transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_sampler = torch.utils.data.distributed.DistributedSampler(testset)

test_loader = torch.utils.data.DataLoader(testset,
                                           batch_size = args.test_bs,
                                               sampler=test_sampler,
                                               pin_memory=True,
                                               drop_last=True,
                                               )

if args.model_path !=None:
    root = "checkpoints/"
    model = ResNet18()
    if args.load_best:
        st_path = root+args.model_path+"/model/"+"best_epoch.pt"
    else:
        st_path = root+args.model_path+"/model/"+"last_epoch.pt"
    st = torch.load(st_path)
    model.load_state_dict(st)
    model = model.to(args.device)
    model.eval()
    
    if args.attack_type == "AA":
        AA = AutoAttack(model, eps=args.eps, version="standard",device=args.device,verbose=False)
        if args.aa_version == "base":
            AA.attacks_to_run = ['apgd-ce','apgd-t']
            
    
    
else:
    print("please give the model path")


if __name__ == '__main__':
    tmp = 0
    total_corr = 0
    total_num = int(args.total_num /args.world_size)
    if args.local_rank == 0:
        test_loader = tqdm(test_loader)
    for images,labels in test_loader:
        tmp += images.size()[0]
        images = images.cuda(non_blocking=True,device=args.device)
        labels = labels.cuda(non_blocking=True,device=args.device)
        
        if args.attack_type == "pgd":
            advs = inf_pgd(model,images,labels,eps=args.eps,step_size=args.step_size,iter_time=args.iter_time)
        
        elif args.attack_type == "AA":
            advs = AA.run_standard_evaluation(images, labels,bs=images.size()[0])
        
        else:
            advs = images
        
        err,corr = acc_test(model,advs,labels)
        total_corr += corr
        #dist.barrier()
        if tmp>=total_num:
            if ddp_available():
                dist.all_reduce(total_corr)
            break
        
    robustness = total_corr/args.total_num
    if args.local_rank == 0:
        print("final robustness:")
        print(robustness)