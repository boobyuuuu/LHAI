"""
Paper:      Enhanced Deep Residual Networks for Single Image Super-Resolution
Url:        https://arxiv.org/abs/1707.02921
Create by:  zh320
Date:       2023/12/16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Activation & ConvAct（保留你之前的多激活支持） ----------
class Activation(nn.Module):
    def __init__(self, act_type='relu', **kwargs):
        super().__init__()
        activation_hub = {
            'relu': nn.ReLU, 'relu6': nn.ReLU6,
            'leakyrelu': nn.LeakyReLU, 'prelu': nn.PReLU,
            'celu': nn.CELU, 'elu': nn.ELU,
            'hardswish': nn.Hardswish, 'hardtanh': nn.Hardtanh,
            'gelu': nn.GELU, 'glu': nn.GLU,
            'selu': nn.SELU, 'silu': nn.SiLU,
            'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax,
            'tanh': nn.Tanh, 'none': nn.Identity,
        }
        act = act_type.lower()
        if act not in activation_hub:
            raise NotImplementedError(f"Unsupported activation: {act_type}")
        self.act = activation_hub[act](**kwargs)

    def forward(self, x):
        return self.act(x)

class ConvAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 groups=1, bias=True, act_type='relu', **kwargs):
        if isinstance(kernel_size, (list, tuple)):
            padding = ((kernel_size[0] - 1) // 2 * dilation,
                       (kernel_size[1] - 1) // 2 * dilation)
        else:
            padding = (kernel_size - 1) // 2 * dilation
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            Activation(act_type, **kwargs)
        )

# ---------- Helper conv ----------
def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True)

# ---------- Residual Block used in EDSR ----------
class EDSRResidualBlock(nn.Module):
    """
    two convs (conv-relu-conv) without BatchNorm, with optional residual scaling.
    """
    def __init__(self, channels, act_type='relu', res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            ConvAct(channels, channels, kernel_size=3, act_type=act_type),
            conv3x3(channels, channels)
        )

    def forward(self, x):
        res = self.body(x)
        if self.res_scale is not None and self.res_scale != 1.0:
            res = res * self.res_scale
        return x + res

# ---------- EDSR (scale=1) model ----------
class EDSR(nn.Module):
    """
    EDSR adapted for LHAI: no upscaling (input/output size unchanged).
    Keep signature compatible: forward(x) -> (output, 0, 0)
    Args:
        in_channels (int): input channels (1 for your dataset)
        out_channels (int): output channels (1)
        num_blocks (int): number of residual blocks (paper uses 32 for large models; can be smaller)
        num_feats (int): feature channels (paper used 64/256 variants; 64 is common)
        act_type (str): activation type passed into ConvAct
        res_scale (float): residual scaling factor (paper used 0.1 for stability)
    """
    def __init__(self, in_channels=1, out_channels=1, num_blocks=16, num_feats=64,
                 act_type='relu', res_scale=0.1):
        super().__init__()
        # head
        self.head = conv3x3(in_channels, num_feats)

        # body: many residual blocks
        body = []
        for _ in range(num_blocks):
            body.append(EDSRResidualBlock(num_feats, act_type=act_type, res_scale=res_scale))
        self.body = nn.Sequential(*body)

        # mid conv (paper: another conv before adding long skip)
        self.mid = conv3x3(num_feats, num_feats)

        # tail: final conv to map features to output channels (no upsampling)
        self.tail = conv3x3(num_feats, out_channels)

        # initialization (kaiming) for conv layers
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Input: x (N, C, H, W) — H=W=64 in your dataset.
        Output: (sr, 0, 0) to be compatible with existing pipeline.
        """
        x_head = self.head(x)
        res = self.body(x_head)
        res = self.mid(res)
        res = res + x_head  # long skip
        out = self.tail(res)
        return out, 0, 0
