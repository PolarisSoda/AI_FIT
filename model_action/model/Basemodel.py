import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel():
    def __init__(self, save_path: str, batch_size: int = 64, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.save_path = save_path

    @abstractmethod
    def train_one_epoch(self):
        pass
    
    @abstractmethod
    def train_with_num_epochs(self, num_epoch: int = 10):
        pass
    
    @torch.no_grad()
    def validate(self):
        return self._validate_impl()
    
    @abstractmethod
    def _validate_impl(self):
        pass

    @abstractmethod
    def save_checkpoint(self, epoch: int):
        pass
    
    @abstractmethod
    def train_with_epoch(self,num_epoch: int) -> None:
        pass

