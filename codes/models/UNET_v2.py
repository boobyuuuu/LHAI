# UNET.py
import torch
import torch.nn as nn

try:
    import deeplay as dl
except Exception as e:
    dl = None
    _IMPORT_ERR = e


class UNET_v2(nn.Module):
    """
    轻量包装：将 dl.AttentionUNet 封装为 LHAI 统一接口
    - __init__ 参数与原始 main 中一致
    - 提供 build() 以保持一致调用
    - forward(x, t) 直接转发给底层 AttentionUNet
    """

    def __init__(
        self,
        jpt=None,  # 兼容占位
        in_channels: int = 2,
        channels=(32, 64, 128),
        base_channels=(256, 256),
        channel_attention=(False, False, False),
        out_channels: int = 1,
        position_embedding_dim: int = 256,
    ):
        super().__init__()
        if dl is None:
            raise ImportError(
                "deeplay 未能导入，请确认已正确安装/可用。\n原始错误：{}".format(_IMPORT_ERR)
            )

        # 保持与你的原始构造完全一致
        self.net = dl.AttentionUNet(
            in_channels=in_channels,
            channels=list(channels),
            base_channels=list(base_channels),
            channel_attention=list(channel_attention),
            out_channels=out_channels,
            position_embedding_dim=position_embedding_dim,
        )

        # 某些版本需要 build() 才完成子模块构建
        # 为了与 main 一致，这里不自动 build，让外部显式调用
        self._built = False

    def build(self):
        """与原 main 中一致的构建入口。"""
        if hasattr(self.net, "build"):
            self.net.build()
        self._built = True
        return self

    def forward(self, x, t=None):
        """
        训练/采样阶段调用方式与原始一致：
        - x: 模型输入（如 [LR | x_t] 拼接）
        - t: 时间步嵌入（sinusoidal 或 learnable），shape ~ (B, D)
        """
        # 兼容两种写法：unet(x, t) / unet(x=x, t=t)
        try:
            return self.net(x=x, t=t)
        except TypeError:
            # 若底层定义是 net(x, t) 位置参数，也能兼容
            return self.net(x, t)
