import torch.nn as nn
import torch

class MLPHead(nn.Module):
    def __init__(self, num_channel: int, num_hidden: int, dropout: float, num_class: int, num_cond_list: list[int]):
        super().__init__()

        self.num_channel = num_channel
        self.num_hidden = num_hidden
        self.dropout = dropout

        self.cls_head = nn.Sequential(
            nn.Linear(num_channel,num_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden,num_class)
        )
        self.cond_head = nn.ModuleList([self._make_head(out_dim) for out_dim in num_cond_list])

        self.apply(self._init_weights)

    def _make_head(self, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.num_channel,self.num_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.num_hidden,out_dim)
        )
    
    def _init_weights(self,m):
        if isinstance(m, nn.Linear): 
            nn.init.kaiming_uniform_(m.weight,nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self,x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        cls_logit = self.cls_head(x)
        cond_logit_list = [head(x) for head in self.cond_head]
        return cls_logit, cond_logit_list

if __name__ == '__main__':
    print("This is the code that implements simple head.")
