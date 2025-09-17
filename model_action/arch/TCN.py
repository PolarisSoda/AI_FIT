import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tcn import TCN
from .Heads import MLPHead

class TCNNetwork(nn.Module):
    def __init__(self, num_class: int = 41, dropout: float = 0.2, cond_counts: list[int] | None = None):
        super().__init__()
        self.num_class = num_class
        self.dropout = dropout

        if cond_counts is None: cond_counts = [5] * self.num_class
        assert len(cond_counts) == self.num_class, "Length mismatch with num_class and cond_counts"
        max_conds = max(cond_counts)

        # Backbone to identify Features
        self.backbone = TCN(
            num_inputs=48,
            num_channels=[64,64,64,64],
            kernel_size=3,
            dropout=self.dropout,
            causal=False,
            use_norm='weight_norm',
            activation='relu',
            input_shape='NLC',
            output_projection=None,
            output_activation=None
        )
        
        self.head = MLPHead(64, 256, 0.1, 41, cond_counts)

    def forward(self, x: torch.tensor):
        feat = self.backbone(x)
        feat = feat.mean(dim = 1)
        return self.head(feat)
