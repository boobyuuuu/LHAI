"""
Paper:      Accelerating the Super-Resolution Convolutional Neural Network
Url:        https://arxiv.org/abs/1608.00367
Create by:  zh320
Date:       2023/12/09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ConvAct(nn.Module):
    """
    Conv + Activation block used in the provided code.
    - conv kernel_size is given
    - act_type: 'prelu' or 'relu'
    - num_parameters: used for PReLU num_parameters (channels)
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 act_type: str = 'prelu', num_parameters: Optional[int] = None,
                 padding: Optional[int] = None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        act_type = (act_type or '').lower()
        if act_type == 'prelu':
            # num_parameters default to out_ch when not specified
            p = num_parameters if (num_parameters is not None) else out_ch
            self.act = nn.PReLU(num_parameters=p)
        elif act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported act_type: {act_type}")

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x

class UpsampleIdentity(nn.Module):
    """
    In original FSRCNN this is the deconvolution (to enlarge).
    Here we remove upscaling (scale==1). This module becomes a
    simple conv that maps `in_ch` -> `out_ch` with a large receptive field
    (kernel_size default 9 in original FSRCNN).
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 9):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x)


class FSRCNN(nn.Module):
    """
    FSRCNN adapted for same-size input/output (no upscaling).
    - Keeps FSRCNN's 'feature extraction -> shrinking -> mapping -> expanding -> deconv' structure
      but removes deconvolution upsampling and replaces it with a mapping conv.
    - forward returns (output, 0, 0) to match given CNN interface.
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 # we ignore `upscale` in this adaptation (kept for API compatibility)
                 upscale: int = 1,
                 d: int = 56,
                 s: int = 12,
                 act_type: str = 'prelu'):
        super().__init__()
        # feature extraction (first part)
        self.first_part = ConvAct(in_channels, d, kernel_size=5, act_type=act_type, num_parameters=d)

        # shrinking -> mapping (mid part)
        # original FSRCNN: shrink d->s (1x1), mapping layers s->s (3x3) repeated, expand s->d (1x1)
        mid_layers = []
        mid_layers.append(ConvAct(d, s, kernel_size=1, act_type=act_type, num_parameters=s))
        # here we keep 4 mapping 3x3 layers (you can change depth)
        for _ in range(4):
            mid_layers.append(ConvAct(s, s, kernel_size=3, act_type=act_type, num_parameters=s))
        mid_layers.append(ConvAct(s, d, kernel_size=1, act_type=act_type, num_parameters=d))
        self.mid_part = nn.Sequential(*mid_layers)

        # last part: originally deconvolution to upsample; here map d -> out_channels with large kernel
        self.last_part = UpsampleIdentity(d, out_channels, kernel_size=9)

        # initialize weights similar to common practice (Xavier) for stable training
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.PReLU):
                # keep default initialization for PReLU
                pass

    def forward(self, x):
        # x shape: (N, C, H, W)
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        # match your CNN interface: return (output, 0, 0)
        return x, 0, 0