import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP层
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 窗口内多头自注意力
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # 例如8
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B_, N, C = x.shape  # B_是窗口数 * batch_size，N是窗口内token数，C是embed_dim
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2,0,3,1,4)  # 3 x B_ x heads x N x head_dim

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Swin Transformer Block（不含shift）
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity()  # 可用DropPath实现随机深度，这里先用恒等
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim*mlp_ratio), drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入token长度必须等于H*W"

        shortcut = x
        x = self.norm1(x)

        # 分窗口操作
        x = x.view(B, H, W, C)

        # 按窗口切分
        windows = window_partition(x, self.window_size)  # (num_windows*B, window_size, window_size, C)
        windows = windows.view(-1, self.window_size * self.window_size, C)  # (num_windows*B, window_size*window_size, C)

        # 窗口内自注意力
        attn_windows = self.attn(windows)

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # (B, H, W, C)

        x = x.view(B, H*W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# 窗口分割函数
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)
    return x

# 窗口反转函数
def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H*W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)
    return x

# Residual Swin Transformer Block（RSTB）
class RSTB(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, input_resolution, num_heads, window_size, mlp_ratio)
            for _ in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        shortcut = x
        B, C, H, W = x.shape
        # Swin Transformer处理需要先reshape成tokens
        x = x.flatten(2).transpose(1,2)  # (B, C, H, W) -> (B, H*W, C)
        for blk in self.blocks:
            x = blk(x)
        # 变回特征图格式
        x = x.transpose(1,2).view(B, C, H, W)
        x = self.conv(x)
        return x + shortcut

# 整体SwinIR_LHAI网络
class SwinIR_LHAI(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, img_size=64,
                 embed_dim=64, depths=[2], num_heads=[4],
                 window_size=8, mlp_ratio=4.):
        super().__init__()
        self.shallow = nn.Conv2d(in_ch, embed_dim, kernel_size=3, padding=1)
        self.rstb = RSTB(embed_dim, (img_size, img_size), depths[0], num_heads[0], window_size, mlp_ratio)
        self.reconstruction = nn.Conv2d(embed_dim, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.shallow(x)
        x = self.rstb(x)
        x = self.reconstruction(x)
        return x, 0, 0
