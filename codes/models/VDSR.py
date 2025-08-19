"""
Paper:      Accurate Image Super-Resolution Using Very Deep Convolutional Networks
Url:        https://arxiv.org/abs/1511.04587
Create by:  zh320
Date:       2023/12/16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VDSR(nn.Module):
    """
    VDSR for same-scale super-resolution (no resizing inside the network).
    Paper: "Accurate Image Super-Resolution Using Very Deep Convolutional Networks"
    Authors: Jiwon Kim, Jung Kwon Lee, Kyoung Mu Lee
    arXiv:1511.04587 (CVPR 2016)

    LHAI-Ready:
      - Input/Output shape unchanged (e.g., N x 1 x 64 x 64)
      - Forward returns (out, 0, 0) to match your CNN example
      - Residual learning: out = x + f(x)
    """
    def __init__(self, jpt: int = 0, in_channels: int = 1, out_channels: int = 1,
                 layer_num: int = 20, hid_channels: int = 64):
        super().__init__()
        assert layer_num >= 3, "VDSR typically uses ~20 layers; must be >=3."

        # First conv (no BN in VDSR)
        self.head = nn.Conv2d(in_channels, hid_channels, kernel_size=3, padding=1, bias=True)

        # Middle conv+ReLU blocks
        body = []
        for _ in range(layer_num - 2):
            body.append(nn.Conv2d(hid_channels, hid_channels, kernel_size=3, padding=1, bias=True))
            body.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*body)

        # Last conv (linear reconstruction)
        self.tail = nn.Conv2d(hid_channels, out_channels, kernel_size=3, padding=1, bias=True)

        # Initialization: Kaiming for Conv + zero bias (common, stable)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (N, C, H, W) — already the target size (e.g., 64x64), no scaling inside.
        residual = x
        out = F.relu(self.head(x), inplace=True)
        out = self.body(out)
        out = self.tail(out)
        out = out + residual  # residual learning
        return out, 0, 0
