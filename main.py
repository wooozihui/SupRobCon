import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
import torchvision
from models.resnet import *
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import argparse
from tqdm import tqdm
from utils import *
from ddp_util import *
from torch.utils.tensorboard import SummaryWriter  
from suprobcon import *
from logger import *
from gradualwarmup import *
from simclr import TransformsSimCLR


### general setting ####

parser = argparse.ArgumentParser(description='PyTorch--Adversarial Training')
parser.add_argument('--local_rank',type=int, default=0)
parser.add_argument('--train_bs',type=int, default=1024)
parser.add_argument('--test_bs',type=int, default=250)
parser.add_argument('--training_epoch',type=int, default=200)
parser.add_argument('--linear_evaluation_training_epoch',type=int, default=100)
parser.add_argument('--savepath',type=str,default='checkpoint')
parser.add_argument('--in_d',type=int,default=512)
parser.add_argument('--out_d',type=int,default=128)
parser.add_argument('--scheduler',type=str,default='mutistep')
parser.add_argument('--init_lr',type=float,default=0.1)
parser.add_argument('--save_last_best',type=bool,default=True)
parser.add_argument('--save_parameters',type=bool,default = True)
## end general setting ##
## warmup ##

parser.add_argument('--scale_factor',type=int,default=1)
parser.add_argument('--warm_epoch',type=int,default=10)
#parser.add_argument('--beta_warmup_epoch',type=int,default=0) not use

## end warmup ##

##### para for AT #####

parser.add_argument('--eps',type=float,default=8/255)
parser.add_argument('--step_size',type=float,default=2/255)
parser.add_argument('--iter_time',type=int,default=10)

##### end AT para #####
##### para for SupRobCon ####

#parser.add_argument('--beta',type=float,default=1)
#parser.add_argument('--temperature_rob',type=float,default=0.5)

#parser.add_argument('--reweight',type=float,default=0.5) ## lamda should be (0,1) to balance the robustness-accuracy tradeoff
parser.add_argument('--temperature',type=float,default='0.5')

##### end SupRobCon para ###

### tensorboard path ###
parser.add_argument('--tb_path',type=str,default="/data/wzh777/My_SimCLR/logs")



args = parser.parse_args()


root = "checkpoints/"
model_savepath = root+args.savepath+'/model'
log_savepath = root+args.savepath+'/log'

if args.local_rank == 0:
    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)
    if args.save_parameters:
        os.system("cp ./run.sh "+root+args.savepath+"/run.sh")

init_ddp(args)

tb_path = args.tb_path

writer = SummaryWriter(tb_path)
logger = Logger(tb_path= tb_path)

torch.backends.cudnn.benchmark = True

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

validset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
valid_sampler = torch.utils.data.distributed.DistributedSampler(validset)
test_sampler = torch.utils.data.distributed.DistributedSampler(testset)



train_loader = torch.utils.data.DataLoader(trainset,
                                           batch_size = int(args.train_bs/args.world_size),
                                               sampler=train_sampler,
                                               pin_memory=True,
                                               drop_last=True,
                                               )

valid_loader = torch.utils.data.DataLoader(validset,
                                           batch_size = args.test_bs,
                                               sampler=valid_sampler,
                                               pin_memory=True,
                                               drop_last=True,
                                               )

test_loader = torch.utils.data.DataLoader(testset,
                                           batch_size = args.test_bs,
                                               sampler=test_sampler,
                                               pin_memory=True,
                                               drop_last=True,
                                               )

backbone = ResNet18()
model = SupRobConModel(backbone,
                   feature_d=args.in_d,
                   mlp_d = args.out_d,)
model.init_suprobcon(temperature=args.temperature,world_size=args.world_size)
model.init_pgd(args.eps,args.step_size,args.iter_time)

model = ddp_model_convert(model,args)

optimizer= torch.optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9, weight_decay=5e-4,nesterov=True)

if args.scheduler == 'cosineanneal':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
             optimizer, args.training_epoch, eta_min=0, last_epoch=-1
        )
if args.scheduler == 'mutistep':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.training_epoch*0.75),int(args.training_epoch*0.9)], gamma=0.1)

if args.scheduler == 'onecyclelr':
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.init_lr, pct_start=0.025, 
                                                                 total_steps=int(args.training_epoch))
    
scheduler_warmup = GradualWarmup(optimizer,warm_epoch=args.warm_epoch,scale_factor=args.scale_factor,after_scheduler=scheduler)
    
## amp ##
scaler = torch.cuda.amp.GradScaler()
## end ## 

## logger register ##
items = ['loss','loss_src','loss_ce','acc']
logger.register(*items)
##.end register ##

## for saving model ##
best_rob = torch.tensor(0)
## end  ####

for epoch in range(args.training_epoch+args.warm_epoch):
    train_sampler.set_epoch(epoch)
    # not use now . model.module.update_epoch(epoch)
    if args.local_rank == 0:
        print("epoch:",epoch)
        loader = tqdm(train_loader)
    else:
        loader = train_loader
    model.train()
    for images,labels in loader:
        images = images.cuda(non_blocking=True,device=args.device)
        #image_1 = images[1].cuda(non_blocking=True,device=args.device)
        labels = labels.cuda(non_blocking=True,device=args.device)
        ## amp ##
        with torch.cuda.amp.autocast():
        ## end ##
            loss,loss_suprobcon,loss_ce_detach = model(images,labels)
            optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        ## amp ##
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ## end ##
    scheduler_warmup.step()
    cur_lr=optimizer.param_groups[-1]['lr']

    #test_acc = linear_evaluation(model,valid_loader,test_loader,args)
    model.eval()
    
    total_num = 1000
    
    train_acc = classifier_test(model,valid_loader,args,test_pic_num=total_num/args.world_size,using_pgd=False)
    test_acc = classifier_test(model,test_loader,args,test_pic_num=total_num/args.world_size,using_pgd=False)
    adv_train_acc = classifier_test(model,valid_loader,args,test_pic_num=total_num/args.world_size,using_pgd=True)
    adv_test_acc = classifier_test(model,test_loader,args,test_pic_num=total_num/args.world_size,using_pgd=True)
    
    if args.local_rank == 0:
        acc = {'normal train':train_acc,
                'normal test':test_acc,
                'adv train': adv_train_acc,
                'adv test': adv_test_acc }
        
        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar("loss/src",loss_suprobcon,epoch)
        writer.add_scalar('loss/ce', loss_ce_detach, epoch)
        #writer.add_scalar('loss/loss_ce_detach', loss_ce_detach, epoch)

        writer.add_scalar("learning rate",cur_lr,epoch)
        #writer.add_scalar("beta",beta,epoch)

        writer.add_scalars('acc', acc, epoch)
        
        logger.update(loss.detach().cpu(),loss_suprobcon.detach().cpu(),loss_ce_detach.detach().cpu(),acc)
        if args.save_last_best:
            if adv_test_acc >= best_rob:
                torch.save(model.module.backbone.state_dict(),os.path.join(model_savepath,"best_epoch.pt"))
                best_rob = adv_test_acc
            torch.save(model.module.backbone.state_dict(),os.path.join(model_savepath,"last_epoch.pt"))
            
        else:
            torch.save(model.module.backbone.state_dict(),os.path.join(model_savepath,str(epoch)+".pt"))
        logger.save(os.path.join(log_savepath,"log.pt"))
        print("loss:",loss)
        print('normal train: ',train_acc,
              ' normal test: ',test_acc,
              ' adv train: ',adv_train_acc,
              ' adv test: ', adv_test_acc)
    