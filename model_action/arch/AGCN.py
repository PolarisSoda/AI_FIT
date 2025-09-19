import torch
import torch.nn as nn
import math
from .Heads import MLPHead
from .utils import *
from torch.autograd import Variable
import numpy as np
from .graph.mygraph import Graph

class unit_tcn(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size,1), padding=(pad,0), stride=(stride,1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        conv_init(self.conv)
        bn_init(self.bn,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))

# ST-GCN with adaptive adjacency(PA).
class unit_gcn(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray, coff_embedding: int = 4, num_subset: int = 3):
        super().__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.num_subset = num_subset

        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32))) # Learnable adjacency (adaptive graph)
        self.register_buffer("A",torch.from_numpy(A.astype(np.float32)))

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for _ in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(dim=-2)
        self.relu = nn.ReLU(inplace=True)

        nn.init.constant_(self.PA, 1e-6)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.to(x.device) + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N,V,V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)                            # N,CT,V
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z if y is None else (y + z)

        y = self.bn(y)
        y = y + self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        return self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))


# -------------------------
# Backbone that returns features (256-d)
# -------------------------
class AGCNBackbone(nn.Module):
    def __init__(self, num_point=25, num_person=1, graph=None, graph_args=dict(), in_channels=3, drop_out=0.0):
        super().__init__()
        if graph is None:
            raise ValueError("graph class path is required")
        
        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        A = self.graph.A

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(in_channels, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.drop = nn.Dropout(drop_out) if drop_out and drop_out > 0 else nn.Identity()
        bn_init(self.data_bn, 1)

    def forward(self, x):
        """
        x: [N, C, T, V, M]
        return: feature [N, 256]
        """
        N, C, T, V, M = x.size()

        # (N,M,V,C,T) → BN over (M*V*C, T)
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x); x = self.l2(x); x = self.l3(x); x = self.l4(x)
        x = self.l5(x); x = self.l6(x); x = self.l7(x)
        x = self.l8(x); x = self.l9(x); x = self.l10(x)

        # Global average pool over T,V and mean over persons M
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1).mean(-1).mean(1)  # [N, 256]
        x = self.drop(x)
        return x

class MultiHeadAGCN(nn.Module):
    def __init__(self,
        num_exercises: int,
        num_states_per_exercise: list,
        num_point: int = 24,
        num_person: int = 1,
        graph: str = "model-action.arch.graph.mygraph.Graph",
        graph_args: dict = {"labeling_mode": "spatial"},
        in_channels: int = 2,
        drop_out: float = 0.5,
        **kwargs):
        super().__init__()

        self.backbone = AGCNBackbone(
            num_point=num_point,
            num_person=num_person,
            graph=graph, graph_args=graph_args,
            in_channels=in_channels,
            drop_out=drop_out)
        
        self.head = MLPHead(
            num_channel=256,
            num_hidden=256,
            dropout=0.2,
            num_class=num_exercises,
            num_cond_list=num_states_per_exercise)

    def forward(self, x):
        """
        x: [N, C, T, V, M]
        returns:
          exercise_logits: [N, num_exercises]
          state_logits_list: List[Tensor] where state_logits_list[i] is [N, n_states_i]
        """
        feat = self.backbone(x)
        return self.head(feat)