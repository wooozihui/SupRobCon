"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
"""

Adjusted by Zihui Wu (zihui@stu.xidian.edu.cn)
Date: 2021/12/15

"""
import torch
import torch.nn as nn
import diffdist
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07,world_size=1,l2norm=True,reweight=None):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.world_size = world_size
        self.l2norm = l2norm
        self.reweight = reweight
    
    
    def reweight_mask(self,mask):
        if self.reweight != None:
            bs = int(mask.size()[0]/3)
            mask[0:2*bs,2*bs:] *= self.reweight
            mask[2*bs:,0:2*bs] *= self.reweight
        return mask
    
    def re_temp(self,mat):
        device = mat.device
        temp_mat = torch.ones_like(mat).to(device)
        temp_mat = temp_mat*self.temperature
        if self.reweight != None:
            bs = int(temp_mat.size()[0]/3)
            temp_mat[0:2*bs,2*bs:] *= self.reweight
            temp_mat[2*bs:,0:2*bs] *= self.reweight
        mat = mat / temp_mat
        return mat
        
    
    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if self.l2norm:
            features = F.normalize(features, dim=2)
        
        if self.world_size > 1:
            features_list = [torch.zeros_like(features) for _ in range(self.world_size)]
            features_list = diffdist.functional.all_gather(features_list, features)
            features = torch.cat(features_list,dim=0)
            
            if labels != None:
                label_lis = [torch.zeros_like(labels) for _ in range(self.world_size)]
                label_lis = diffdist.functional.all_gather(label_lis, labels)
                labels = torch.cat(label_lis,dim=0)
        
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

 
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

if __name__ == '__main__':
    critertion  = SupConLoss(reweight=0.5)
    aaa = torch.randn(4,3,128)
    labels = torch.LongTensor([1,2,1,4])
    loss = critertion(aaa,labels)
    