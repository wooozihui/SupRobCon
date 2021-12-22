import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
import torch.distributed as dist
import argparse

from utils import inf_pgd


def init_ddp(args):
    args.world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(
        'nccl',
        init_method='env://',
        rank = args.local_rank,
        world_size = args.world_size
    )
    args.device = "cuda:"+str(args.local_rank)
    args.using_sync_bn=True

def ddp_model_convert(model,args):
    model = model.to(args.device)
    if args.using_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(args.device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],find_unused_parameters=False)
    return model

def ddp_available():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def acc_test(model,image,label):
    batchsize = image.size()[0]
    logits = model(image)
    label = label.view(-1)
    max_pos = torch.argmax(logits,dim=1)
    corr = (label == max_pos).int().sum()
    err = batchsize - corr
    return err,corr

def classifier_test(model,testloader,args,test_pic_num=1000,using_pgd=False):
    right = 0
    total = 0
    for i,(image,label) in enumerate(testloader):
        image = image.to(args.device)
        label = label.to(args.device)
        if using_pgd:
            image = inf_pgd(model.module.backbone,image,label,20,eps=8/255,step_size=2/255)
        batchsize = image.size()[0]
        err0,right0 = acc_test(model.module.backbone,image,label)
        right+=right0
        total+=batchsize
        tested = (i+1)*batchsize
        if tested >= test_pic_num:
            break
    if ddp_available():
        dist.all_reduce(right)
        total = total * args.world_size
    return right/total
    
