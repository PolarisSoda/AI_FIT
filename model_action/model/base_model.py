import torch
import torch.nn as nn

class BaseModel():
    def __init__(self,opt):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_one_epoch(self):
        pass

    def load_pretrain():
        pass
    
    @torch.no_grad
    def inference(self, input: torch.Tensor):
        pass
    
    def save_model(self):
        pass 
    
    def train_with_epoch(self,num_epoch: int) -> None:
        pass

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        pass

    def model_to_device(self,net):
        net = net.to(self.device)

    def get_optimizer(self,optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params,lr,**kwargs)
        elif optim_type == 'Adamw':
            optimizer = torch.optim.Adamw(params,lr,**kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params,lr,**kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
    
    def setup_schedulers(self):
        pass

