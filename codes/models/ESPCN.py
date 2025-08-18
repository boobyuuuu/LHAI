"""
Paper:      Real-Time Single Image and Video Super-Resolution Using an Efficient 
            Sub-Pixel Convolutional Neural Network
Url:        https://arxiv.org/abs/1609.05158
Create by:  zh320
Date:       2023/12/09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Activation(nn.Module):
    """
    通用激活封装：保留原仓库的风格，同时避免不必要的耦合。
    注意：Softmax/GLU 等需要 dim 的激活，若要使用请在实例化时传入 dim=...
    """
    def __init__(self, act_type: str = "prelu", **kwargs):
        super().__init__()
        hub = {
            'relu': nn.ReLU, 'relu6': nn.ReLU6,
            'leakyrelu': nn.LeakyReLU, 'prelu': nn.PReLU,
            'celu': nn.CELU, 'elu': nn.ELU,
            'hardswish': nn.Hardswish, 'hardtanh': nn.Hardtanh,
            'gelu': nn.GELU, 'selu': nn.SELU, 'silu': nn.SiLU,
            'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh,
            'softmax': nn.Softmax, 'glu': nn.GLU,
            'none': nn.Identity
        }
        act_type = (act_type or 'none').lower()
        if act_type not in hub:
            raise NotImplementedError(f'Unsupported activation type: {act_type}')
        self.act = hub[act_type](**kwargs)

    def forward(self, x):
        return self.act(x)


class ConvAct(nn.Sequential):
    """
    卷积 + 激活 的小积木：默认 same padding，保持空间尺寸不变
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1,
                 groups=1, bias=True, act_type='prelu', **act_kwargs):
        if isinstance(kernel_size, (list, tuple)):
            padding = ((kernel_size[0] - 1) // 2 * dilation,
                       (kernel_size[1] - 1) // 2 * dilation)
        else:
            padding = (kernel_size - 1) // 2 * dilation
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias),
            Activation(act_type, **act_kwargs)
        )


class ESPCN(nn.Module):
    """
    LHAI-ESPCN (scale=1)
    - 不改变输入/输出大小 (B, 1, 64, 64) -> (B, 1, 64, 64)
    - 保留 ESPCN 的轻量特性：5x5 -> 3x3 -> 线性 3x3
    - 中间层保留 Activation；输出层默认无激活（可配置）

    返回: (y, 0, 0) 以兼容 LHAI 框架
    """
    def __init__(self,
                 jpt=None,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 n1: int = 64,
                 n2: int = 32,
                 act_type: str = 'prelu',
                 out_act_type: str = 'none'):
        super().__init__()

        # 表示学习（大核 5x5）
        self.layer1 = ConvAct(in_channels, n1, kernel_size=5, act_type=act_type)
        # 细化表示（3x3）
        self.layer2 = ConvAct(n1, n2, kernel_size=3, act_type=act_type)
        # 线性重建头（3x3）
        self.recon = nn.Conv2d(n2, out_channels, kernel_size=3, padding=1, bias=True)
        # 可选输出激活（默认 none，便于实值回归）
        self.out_act = Activation(out_act_type)

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.recon(y)
        y = self.out_act(y)
        return y, 0, 0
