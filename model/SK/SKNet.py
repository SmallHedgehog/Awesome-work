""" Reference:
    [1] Xiang Li etc. Selective Kernel Networks. CVPR 2019. https://arxiv.org/abs/1903.06586
"""

import torch.nn as nn
import torch

from functools import reduce


class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, G=32, r=16, L=32):
        """
        Parameters
        ----------
        in_channels (int): input channels dimensionality.
        out_channels (int): output channels dimensionality.
        stride (int): stride.
        M (int): the number of different kernels to be aggregated.
        G (int): the cardinality of each path.
        r (int): reduction ratio that controls the number of parameters.
        L (int): the minimum dim of the Excitation operator.
        """
        super(SKConv, self).__init__()

        self.reduce = max(out_channels // r, L)

        self.convs = nn.ModuleList()
        for idx in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1 + idx,
                          dilation=1 + idx, groups=G, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))

        self.GAP = nn.AdaptiveAvgPool2d(output_size=1)
        self.softmax = nn.Softmax(dim=-1)

        self.fc = nn.Linear(out_channels, self.reduce, bias=False)
        self.fcs = nn.ModuleList()
        for idx in range(M):
            self.fcs.append(nn.Linear(self.reduce, out_channels, bias=False))

    def forward(self, x):
        feats = [conv(x) for conv in self.convs]

        sum_feats = reduce(lambda x, y: x + y, feats)
        agp_feats = self.GAP(sum_feats).squeeze()

        Z_feats = self.fc(agp_feats)

        attention_feats = [fc(Z_feats) for fc in self.fcs]

        attention_feats = torch.cat([feat.unsqueeze(-1) for feat in attention_feats], dim=2)
        soft_attention = self.softmax(attention_feats).unsqueeze(2)
        soft_attention = soft_attention.unsqueeze(2)

        feats = torch.cat([feat.unsqueeze(-1) for feat in feats], dim=-1)
        return torch.sum(feats * soft_attention, dim=-1)
