import torch

class GradualWarmup():
    def __init__(self,optim,warm_epoch,scale_factor,after_scheduler):
        self.warm_epoch = warm_epoch
        self.scale_factor = scale_factor
        self.after_scheduler = after_scheduler
        self.last_epoch = 0
        self.optim = optim
        init_lr = optim.param_groups[0]['lr']
        self.init_lr = init_lr
        self.endwarm_lr = scale_factor* init_lr
        
        
    def step(self):
        if self.last_epoch < self.warm_epoch:
            cur_lr = self.init_lr + (self.endwarm_lr-self.init_lr)*((self.last_epoch+1)/self.warm_epoch)
            for param_group in self.optim.param_groups:
                param_group['lr'] = cur_lr
        else:
                self.after_scheduler.step()
            
        self.last_epoch += 1