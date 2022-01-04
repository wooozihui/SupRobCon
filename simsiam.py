import torch
import torch.nn as nn

class SimSiamLoss(nn.Module):
    def __init__(self,):
        self.criterion = nn.CosineSimilarity(dim=1)

    def forward(self,p1,p2,z1,z2):
        z1 = z1.detach()
        z2 = z2.detach()
        loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
        return loss

class SimSiamModel(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, backbone, in_d=512, out_d=2048):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiamModel, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.backbone = backbone
        self.simsiam = SimSiamLoss()
        # build a 3-layer projector
        self.projection = nn.Sequential(nn.Linear(in_d, in_d, bias=False),
                                        nn.BatchNorm1d(in_d),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(in_d, in_d, bias=False),
                                        nn.BatchNorm1d(in_d),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(in_d,out_d,bias=False),
                                        nn.BatchNorm1d(out_d, affine=False)) # output layer

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(out_d, in_d, bias=False),
                                        nn.BatchNorm1d(in_d),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(in_d, out_d)) # output layer

    
    def get_feature(self,x):
        out = F.relu(self.backbone.bn1(self.backbone.conv1(x)))
        out = self.backbone.layer1(out)
        out = self.backbone.layer2(out)
        out = self.backbone.layer3(out)
        out = self.backbone.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
    

    def forward(self, x1, x2,label):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        f1 = self.get_feature(x1)
        f2 = self.get_feature(x2)
        
        z1 = self.projection(f1) # NxC
        z2 = self.projection(f1) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        loss_simsiam = self.simsiam(p1,p2,z1,z2)
        
        f_bk = f1.clone().detach()
        logits = self.backbone.linear(f_bk)
        loss_ce_detach = torch.nn.CrossEntropyLoss()(logits,label)
        
        return loss 