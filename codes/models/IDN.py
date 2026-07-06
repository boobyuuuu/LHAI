"""
Paper:      Fast and Accurate Single Image Super-Resolution via Information Distillation Network
Url:        https://arxiv.org/abs/1803.09454
Create by:  zh320
Date:       2023/12/30
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- 基础模块 ---------
def conv1x1(in_ch, out_ch, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=bias)

class ConvAct(nn.Module):
    """
    统一的 3x3 卷积 + 可选激活；支持 groups 分组卷积
    act_type: 'relu' | 'leakyrelu' | 'prelu' | None
    """
    def __init__(self, in_ch, out_ch, k=3, groups=1, act_type='leakyrelu', bias=True):
        super().__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, groups=groups, bias=bias)
        if act_type is None:
            self.act = nn.Identity()
        elif act_type.lower() == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type.lower() == 'leakyrelu':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif act_type.lower() == 'prelu':
            self.act = nn.PReLU(num_parameters=out_ch)
        else:
            raise ValueError(f'Unsupported act_type: {act_type}')

    def forward(self, x):
        return self.act(self.conv(x))


# --------- IDN 主体 ----------
class EnhancementUnit(nn.Module):
    """
    信息蒸馏单元（IDN 的核心）
    通道按 d / (C-d) 进行分流，实现“逐级蒸馏 + 残差”
    """
    def __init__(self, C, d, act_type='leakyrelu', groups=(1,4,1,4,1,1)):
        super().__init__()
        assert len(groups) == 6, 'Length of groups should be 6.'
        self.d = d

        # 第一阶段: 减通道 -> 再扩回 C
        self.conv1 = nn.Sequential(
            ConvAct(C, C - d, 3, groups=groups[0], act_type=act_type),
            ConvAct(C - d, C - 2 * d, 3, groups=groups[1], act_type=act_type),
            ConvAct(C - 2 * d, C, 3, groups=groups[2], act_type=act_type),
        )

        # 第二阶段: 对“主干”部分再加工，最终输出 C + d 以便与被蒸馏分支拼接
        self.conv2 = nn.Sequential(
            ConvAct(C - d, C, 3, groups=groups[3], act_type=act_type),
            ConvAct(C, C - d, 3, groups=groups[4], act_type=act_type),
            ConvAct(C - d, C + d, 3, groups=groups[5], act_type=act_type),
        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)                     # 回到 C 通道
        x_c = x[:, :self.d, :, :]             # 蒸馏分支（保留 d 通道）
        x_c = torch.cat([x_c, residual], 1)   # 与残差拼接，C + d
        x_s = x[:, self.d:, :, :]             # 主干分支
        x_s = self.conv2(x_s)                 # 输出 C + d
        return x_s + x_c                      # 对齐通道后逐元素相加


class DBlock(nn.Sequential):
    """ 信息蒸馏块：单个 EnhancementUnit 后接 1x1 压回基宽 C """
    def __init__(self, C, d, act_type='leakyrelu'):
        super().__init__(
            EnhancementUnit(C, d, act_type=act_type),
            conv1x1(C + d, C)
        )


class IDN(nn.Module):
    """
    IDN（Information Distillation Network）
    已简化为“等尺度超分辨/去模糊”设置：输入输出同尺寸（64x64），无上采样分支。
    与示例 CNN 接口保持一致：
        - __init__(jpt=0)
        - forward(x) -> (y, 0, 0)
    """
    def __init__(self, jpt: int = 0,
                 in_channels: int = 1, out_channels: int = 1,
                 num_blocks: int = 4, C: int = 64, s: int = 4,
                 act_type: str = 'leakyrelu'):
        super().__init__()
        assert s > 1, 's must be > 1 for valid split ratio.'
        self.jpt = jpt

        # 蒸馏比例 d = C / s
        d = int(C / s)

        # 特征提取
        self.fblock = nn.Sequential(
            ConvAct(in_channels, C, 3, act_type=act_type),
            ConvAct(C, C, 3, act_type=act_type),
        )

        # 多个信息蒸馏块堆叠
        self.dblocks = nn.Sequential(*[DBlock(C, d, act_type) for _ in range(num_blocks)])

        # 重建层（等尺度，直接卷积到输出通道）
        self.recon = nn.Conv2d(C, out_channels, kernel_size=3, padding=1, bias=True)

        # 轻量初始化（可按需换成更复杂的 init）
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        输入:  (N, 1, H, W)  这里的 H=W=64（按您的数据规定）
        输出:  (N, 1, H, W)  与输入同尺寸
        返回:  与示例 CNN 一致的三元组 (y, 0, 0)
        """
        shallow = self.fblock(x)
        deep = self.dblocks(shallow)
        y = self.recon(deep) + x              # 等尺度残差（对退化/模糊图进行细节回补）
        return y, 0, 0
