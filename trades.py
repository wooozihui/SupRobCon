import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from utils import *

def kl_div(logits,adv_logits):
    bs = logits.size()[0]
    criterion_kl = nn.KLDivLoss(reduction='sum')
    loss_kl = (1.0 / bs) * criterion_kl(F.log_softmax(adv_logits, dim=1),
                                                        F.softmax(logits, dim=1))
    return loss_kl
    
