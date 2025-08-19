"""
Paper:      Image Super-Resolution via Deep Recursive Residual Network
Url:        https://openaccess.thecvf.com/content_cvpr_2017/html/Tai_Image_Super-Resolution_via_CVPR_2017_paper.html
Create by:  zh320
Date:       2023/12/23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_ch, out_ch, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias)

class ResidualUnit(nn.Module):
    """
    两层 3x3 Conv + ReLU 的残差单元。
    作为“共享权重”的递归体被重复调用 U 次（参数共享是 DRRN 的关键）。
    """
    def __init__(self, channels: int, act: str = 'relu'):
        super().__init__()
        self.conv1 = conv3x3(channels, channels)
        self.conv2 = conv3x3(channels, channels)
        if act == 'prelu':
            self.act = nn.PReLU(num_parameters=channels)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.act(self.conv1(x))
        res = self.conv2(res)
        return x + res  # 局部残差

class DRRN(nn.Module):
    """
    等尺度 DRRN（无上采样），适配 LHAI：
    - 输入/输出尺寸不变
    - forward 返回 (out, 0, 0)

    结构：Head → [共享 ResidualUnit 递归 U 次 + 与首份特征的短接] → Tail → 全局残差到输入
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hid_channels: int = 128,
        U: int = 9,              # 递归次数（经典设定 9~25 之间可选）
        act_type: str = 'relu'   # 'relu' 或 'prelu'
    ):
        super().__init__()
        self.U = U

        # Head：把图像提到特征空间
        self.head = conv3x3(in_channels, hid_channels)

        # 共享的 ResidualUnit：被重复调用 U 次（参数共享）
        self.shared_unit = ResidualUnit(hid_channels, act=act_type)

        # Tail：把特征映射回图像域
        self.tail = conv3x3(hid_channels, out_channels)

    def forward(self, x):
        # 保存输入用于全局残差
        x_in = x

        # Head
        feat0 = F.relu(self.head(x), inplace=True)

        # 递归：每次都对同一“共享”单元调用（参数共享）
        # 同时与首次特征 feat0 建立短接（与不少 DRRN 实作一致，增强信息流）
        feat = feat0
        for _ in range(self.U):
            feat = self.shared_unit(feat)
            feat = feat + feat0

        # Tail + 全局残差到输入
        out = self.tail(feat)
        out = x_in + out

        # 适配 LHAI 的 (pred, aux1, aux2) 约定
        return out, 0, 0
