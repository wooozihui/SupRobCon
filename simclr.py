import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import *
import torch.distributed as dist
import diffdist
import torchvision
from ddp_util import *
from tqdm import tqdm
from utils import *
#### this function is from https://github.com/Spijkervet/SimCLR  ####
class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size, train=True):
        s = 1
        self.train = train
        
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        if self.train:
            return self.train_transform(x), self.train_transform(x)
        else:
            return self.test_transform(x)



class SimclrEncoder(nn.Module):
    def __init__(self,backbone,feature_d = 512,mlp_d = 128):
        super(SimclrEncoder, self).__init__()
        self.backbone = backbone
        
        self.mlp_head = nn.Sequential(
            nn.Linear(feature_d, feature_d, bias=False),
            nn.ReLU(),
            nn.Linear(feature_d, mlp_d, bias=False),
        )
    
    def get_feature(self,x):
        out = F.relu(self.backbone.bn1(self.backbone.conv1(x)))
        #print(out.size())
        out = self.backbone.layer1(out)
        #print(out.size())
        out = self.backbone.layer2(out)
        #print(out.size())
        out = self.backbone.layer3(out)
        out = self.backbone.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
    
    def forward(self,x):
        out = self.get_feature(x)
        out = self.mlp_head(out)
        return out

class AddLinearLayer(nn.Module):
    def __init__(self,simmodel,linearlayer):
        super(AddLinearLayer, self).__init__()
        self.simmodel = simmodel
        self.linearlayer = linearlayer
    def forward(self,x):
        out = self.simmodel.get_feature(x)
        out = self.linearlayer(out)
        return out

#### this function is originly from https://github.com/Spijkervet/SimCLR , adjusted by zihui Wu  ####
class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size
        
        #z = torch.cat((z_i, z_j), dim=0)
        z_list_i = [torch.zeros_like(z_i) for _ in range(dist.get_world_size())]
        z_list_j = [torch.zeros_like(z_j) for _ in range(dist.get_world_size())]
        if self.world_size > 1:
            z_list_i = diffdist.functional.all_gather(z_list_i, z_i)
            z_list_j = diffdist.functional.all_gather(z_list_j, z_j)
            
            z_i = torch.cat(z_list_i,dim=0)
            z_j = torch.cat(z_list_j,dim=0)
        z = torch.cat((z_i, z_j), dim=0)
        
        ## when training resnet50, we find normalize is better ##
        z = F.normalize(z, p=2, dim=1)


        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        
        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

#####   linear evaluation (only training the last linear layer and freeze   ######
#####   the backbone which is training with the unsupervised fashion)       ######
def linear_evaluation(model,train_loader,test_loader,args):
    '''
    model: the simclr model which is defined in this file as the class SimclrEncoder
    train_loader: ddp train dataloader
    test_loader: ddp test dataloader
    train_bs: training batch size (not use)
    test_bs: test batch size (not use)
    training_epoch: training epochs before testing
    '''
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    features = []
    labels = []
    training_epoch = args.linear_evaluation_training_epoch
    with torch.no_grad():
        for image,label in train_loader:
            image = image.to(args.device)
            feature = model.module.get_feature(image)
            features.append(feature.detach().cpu())
            labels.append(label)
    #features = torch.cat(features,dim=0)
    
    linearlayer = nn.Linear(args.in_d, args.out_d)
    linearlayer = linearlayer.to(args.device)
    optimizer= torch.optim.SGD(linearlayer.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-6)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #        optimizer, training_epoch, eta_min=0, last_epoch=-1
    #    )
    ### training ###
    if args.local_rank == 0:
        print("linear evaluation training start")
        epoch_bar = tqdm(range(training_epoch))
    else:
        epoch_bar = range(training_epoch)
    for epoch in epoch_bar:
        adjust_learning_rate(optimizer,epoch,init_lr=0.1,schedule='cosine')
        for feature,label in zip(features,labels):
            feature = feature.to(args.device)
            label = label.to(args.device)
            logits= linearlayer(feature)
            loss = criterion(logits,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #scheduler.step()

    if args.local_rank == 0:
        print("training end, start testing")
    final_model = AddLinearLayer(model.module,linearlayer)
    total_num = 10000
    with torch.no_grad():
        test_acc = classifier_test(final_model,test_loader,args,test_pic_num=total_num/args.world_size,using_pgd=False)
    if args.local_rank == 0:
        print("final acc :",test_acc)
    return test_acc

    

