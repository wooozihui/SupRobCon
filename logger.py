import torch
from torch.utils.tensorboard import SummaryWriter  
class Logger():
    def __init__(self,tb_path='/data/wzh777/My_SimCLR/logs'):
        self.max_epoch = 0
        self.tb_path = tb_path
        self.writer = SummaryWriter(tb_path)
        self.item_size = 0
        self.epoch_data_list = []
    
    ## register: should be done before update ##
    def register(self,*item):
        self.items = item
        self.item_size+=len(item)
    
    ## save data as dict ##
    def update(self,*data):
        epoch_dict = {}
        for i in range(self.item_size):
            epoch_dict[self.items[i]] = data[i]
        self.epoch_data_list.append(epoch_dict)
        self.max_epoch+=1
        
    def get_data_epoch(self,item,epoch):
        epoch_dict = self.epoch_data_list[epoch]
        data = epoch_dict[item]
        return data
    
    def save(self,path):
        torch.save({'item_size':self.item_size,'items':self.items,'epoch_data_list':self.epoch_data_list},path)
        
    def resume(self,path):
        checkpoint = torch.load(path)
        self.item_size = checkpoint['item_size']
        self.items = checkpoint['items']
        self.epoch_data_list = checkpoint['epoch_data_list']
        self.max_epoch = len(self.epoch_data_list)
        
    def allwirte2tb(self):
        for epoch,items in enumerate(self.epoch_data_list):
            for item in items:
                if isinstance(items[item],dict):
                    self.writer.add_scalars(item,items[item], epoch)
                else:
                    self.writer.add_scalar(item,items[item], epoch)
        self.writer = SummaryWriter(self.tb_path)

## analyser class : from epoch data get the information ##
class Analyser():
    def __init__(self,logger):
        self.logger = logger
        self.max_epoch = logger.max_epoch
        
    ## roubst overfitting analysis (according to Rice, this metirc ##
    ## should be the diff of the best robust accuracy and the last ##
    ## robust accuracy)                                            ##
    def robust_overfitting_ana(self):
        ## firstly, we should locate the best rob acc epoch ##
        ## note that the epoch is counting form zero   ##
        last_rob = 0
        best_rob = 0
        best_epoch = 0
        last_epoch = self.max_epoch-1
        for epoch in range(self.max_epoch):
            acc_dict = self.logger.get_data_epoch('acc',epoch)
            test_rob = acc_dict['adv test']
            if epoch == self.max_epoch-1:
                last_rob = test_rob
            if test_rob >= best_rob:
                best_rob = test_rob
                best_epoch = epoch
        diff = best_rob - last_rob
        
        return diff,best_epoch,last_epoch,best_rob,last_rob

class Meaner():
    def __init__(self):
        pass
    def update(self):
        pass
    
if __name__ == '__main__':
    logger = Logger()
    items = ['test1','test2']
    logger.register(*items)
    print(logger.items)
    logger.update(1,2)
    data = logger.get_data_epoch('test1',0)
    print(data)
    logger.update(3,4)
    data = logger.get_data_epoch('test2',1)
    print(data)
    print(logger.epoch_data_list)




    
                    