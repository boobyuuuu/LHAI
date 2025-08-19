"""
Paper:      Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution
Url:        https://arxiv.org/abs/1704.03915
Create by:  zh320
Date:       2023/12/16
"""
"""
Paper:      Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution
Url:        https://arxiv.org/abs/1704.03915
Create by:  zh320
Date:       2023/12/16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class _ConvAct(nn.Module):
    """Conv + LeakyReLU（可关）"""
    def __init__(self, in_ch, out_ch, k=3, act=True):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True) if act else None
        # Kaiming 初始化；最后输出层会单独零初始化，做“从恒等开始学习”
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='leaky_relu')
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        return self.act(x) if self.act is not None else x


class LapSRN(nn.Module):
    """
    LapSRN (no-scale, same-size I/O) for LHAI
    - 与 LHAI 的 CNN 接口保持一致：forward(x) -> (y, 0, 0)
    - 单尺度（不放大），保留 LapSRN 的“深特征提取 + 残差重建”思想：
        trunk: 多层卷积抽特征
        head_out: 预测残差（无激活）
        y = x + residual
    """
    def __init__(self, jpt: int = 0,
                 in_channels: int = 1, out_channels: int = 1,
                 hid_channels: int = 64, fe_layers: int = 10):
        super().__init__()
        assert fe_layers >= 2, "fe_layers 至少为 2（含首层与尾层之间的若干中间层）。"

        layers = []
        # 首层
        layers.append(_ConvAct(in_channels, hid_channels, k=3, act=True))
        # 中间特征堆叠
        for _ in range(fe_layers - 2):
            layers.append(_ConvAct(hid_channels, hid_channels, k=3, act=True))
        self.trunk = nn.Sequential(*layers)

        # 输出层（预测残差）：无激活，且零初始化 → 从“恒等映射”开始学习
        self.head_out = _ConvAct(hid_channels, out_channels, k=3, act=False)
        nn.init.zeros_(self.head_out.conv.weight)
        nn.init.zeros_(self.head_out.conv.bias)

    def forward(self, x):
        feat = self.trunk(x)
        residual = self.head_out(feat)
        y = x + residual  # Laplacian 残差重建（同尺寸）
        return y, 0, 0
