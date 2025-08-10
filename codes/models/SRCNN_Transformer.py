import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Activation & ConvAct ----------
class Activation(nn.Module):
    def __init__(self, act_type='relu', **kwargs):
        super().__init__()
        activation_hub = {
            'relu': nn.ReLU, 'prelu': nn.PReLU, 'leakyrelu': nn.LeakyReLU,
            'gelu': nn.GELU, 'silu': nn.SiLU, 'none': nn.Identity
        }
        act = act_type.lower()
        if act not in activation_hub:
            raise NotImplementedError(f"Unsupported activation: {act_type}")
        self.act = activation_hub[act](**kwargs)

    def forward(self, x):
        return self.act(x)

class ConvAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, groups=1, bias=True, act_type='relu', **kwargs):
        if isinstance(kernel_size, (list, tuple)):
            pad = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        else:
            pad = (kernel_size - 1) // 2 * dilation
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding=pad, dilation=dilation, groups=groups, bias=bias),
            Activation(act_type, **kwargs)
        )

# ---------- Patch Transformer (local patch + TransformerEncoder) ----------
class PatchTransformer(nn.Module):
    """
    Extract non-overlapping patches with Unfold -> linear project -> TransformerEncoder (batch_first)
    -> linear project back -> Fold to feature map.
    This implements a localized transformer block (Swin-like behavior without shift).
    """
    def __init__(self, in_channels, patch_size=4, embed_dim=64, num_layers=2, nhead=4, dim_feedforward=None, dropout=0.0):
        super().__init__()
        assert patch_size >= 1 and isinstance(patch_size, int)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_dim = in_channels * (patch_size ** 2)
        self.embed_dim = embed_dim

        # linear projections
        self.proj_in = nn.Linear(self.patch_dim, embed_dim)
        self.proj_out = nn.Linear(embed_dim, self.patch_dim)

        # TransformerEncoder
        if dim_feedforward is None:
            dim_feedforward = embed_dim * 4
        # try to use batch_first if supported; fallback to seq-first
        try:
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
                                                       dim_feedforward=dim_feedforward, dropout=dropout,
                                                       batch_first=True)
            self._batch_first = True
        except TypeError:
            # older torch: no batch_first
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
                                                       dim_feedforward=dim_feedforward, dropout=dropout)
            self._batch_first = False

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # unfold / fold will be created on forward since H/W unknown at init

    def forward(self, x):
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, f"Height/Width must be divisible by patch_size ({p}). Got H={H}, W={W}"

        unfold = nn.Unfold(kernel_size=p, stride=p)
        fold = nn.Fold(output_size=(H, W), kernel_size=p, stride=p)

        patches = unfold(x)           # (N, C*p*p, L) where L = (H/p)*(W/p)
        patches = patches.permute(0, 2, 1)  # (N, L, patch_dim)

        tokens = self.proj_in(patches)      # (N, L, embed_dim)

        # Transformer expects (N, L, E) if batch_first else (L, N, E)
        if self._batch_first:
            tokens = self.transformer(tokens)   # (N, L, E)
        else:
            tokens = tokens.permute(1, 0, 2)    # (L, N, E)
            tokens = self.transformer(tokens)
            tokens = tokens.permute(1, 0, 2)    # back to (N, L, E)

        patches_rec = self.proj_out(tokens)   # (N, L, patch_dim)
        patches_rec = patches_rec.permute(0, 2, 1)  # (N, patch_dim, L)
        out = fold(patches_rec)               # (N, C, H, W)
        return out

# ---------- Upsampler: conv -> PixelShuffle ----------
class PixelShuffleUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super().__init__()
        self.scale = scale
        if scale == 1:
            # identity mapping (just conv)
            self.body = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            # conv to expand channels: out_channels * (scale^2)
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * (scale ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(scale)
            )

    def forward(self, x):
        return self.body(x)

# ---------- Full SRCNN + Transformer model ----------
class SRCNN_Transformer(nn.Module):
    """
    SRCNN backbone replaced by local Patch-Transformer blocks + PixelShuffle upsampling.
    Forward returns (output, 0, 0) to be compatible with LHAI.

    Args:
        jpt: placeholder first arg for compatibility with other models (can be None)
        in_channels, out_channels: image channels
        scale: upsampling factor (set 1 to keep spatial size unchanged)
        shallow_feats: feature channels after first conv
        patch_size: patch size for transformer (should divide H and W)
        transformer_layers: number of transformer encoder layers
        transformer_heads: attention heads
        transformer_embed: embedding dimension for transformer tokens
    """
    def __init__(self, jpt=None, in_channels=1, out_channels=1, scale=1,
                 shallow_feats=64, patch_size=4,
                 transformer_layers=2, transformer_heads=4, transformer_embed=64,
                 act_type='relu'):
        super().__init__()

        # shallow feature extraction (replaces SRCNN layer1 & layer2)
        # keep relatively large receptive field like SRCNN first conv
        self.shallow = nn.Sequential(
            ConvAct(in_channels, shallow_feats, kernel_size=9, act_type=act_type),
            ConvAct(shallow_feats, shallow_feats, kernel_size=3, act_type=act_type)
        )

        # patch transformer (operates on feature maps)
        self.transformer = PatchTransformer(in_channels=shallow_feats,
                                            patch_size=patch_size,
                                            embed_dim=transformer_embed,
                                            num_layers=transformer_layers,
                                            nhead=transformer_heads)

        # 1x1 conv (channel projection / dimensionality reduction)
        self.reduce = nn.Conv2d(shallow_feats, shallow_feats, kernel_size=1, padding=0)

        # upsampler (pixelshuffle)
        self.upsampler = PixelShuffleUpsampler(shallow_feats, out_channels, scale)

        # initialize convs (kaiming)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (N, C, H, W)
        returns: (sr, 0, 0)
        """
        # shallow features
        feat = self.shallow(x)        # (N, F, H, W)

        # transformer operates on same spatial resolution but models long-range relation in local patches
        feat_tr = self.transformer(feat)  # (N, F, H, W)

        # combine and reduce
        feat_comb = self.reduce(feat + feat_tr)  # skip connection between shallow conv and transformer

        # upsample
        out = self.upsampler(feat_comb)

        return out, 0, 0
