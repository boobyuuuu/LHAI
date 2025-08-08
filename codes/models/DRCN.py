"""
Paper:      Deeply-Recursive Convolutional Network for Image Super-Resolution
Url:        https://arxiv.org/abs/1511.04491
Create by:  zh320
Date:       2023/12/23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Activation（保留你给的完整实现） =====
class Activation(nn.Module):
    def __init__(self, act_type, **kwargs):
        super(Activation, self).__init__()
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

        act_type = act_type.lower()
        if act_type not in activation_hub:
            raise NotImplementedError(f'Unsupport activation type: {act_type}')

        # Some activations accept params via kwargs (e.g., Softmax requires dim)
        self.activation = activation_hub[act_type](**kwargs)

    def forward(self, x):
        return self.activation(x)

# ===== ConvAct（保留原实现） =====
class ConvAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 groups=1, bias=True, act_type='relu', **kwargs):
        if isinstance(kernel_size, (list, tuple)):
            padding = ((kernel_size[0] - 1) // 2 * dilation,
                       (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):
            padding = (kernel_size - 1) // 2 * dilation

        super(ConvAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            Activation(act_type, **kwargs)
        )

# ===== DRCN with recursive-supervision + learnable fusion weights =====
class DRCN(nn.Module):
    """
    DRCN with optional recursive-supervision (side outputs + learnable fusion weights).
    Default forward(x) -> (fused_out, 0, 0) to be compatible with your existing pipeline.

    Args:
        in_channels, out_channels: channel dims
        recursions: number of recursive steps (paper uses up to 16)
        hid_channels: hidden channels
        act_type: activation string (passed to Activation)
        arch_type: 'basic' or 'advanced' (advanced uses skip connection)
        use_recursive_supervision: if True, create learnable fusion weights and return side outputs when requested
    """
    def __init__(self, in_channels=1, out_channels=1,
                 recursions=16, hid_channels=256, act_type='relu',
                 arch_type='advanced', use_recursive_supervision=False):
        super(DRCN, self).__init__()
        if arch_type not in ['basic', 'advanced']:
            raise ValueError(f'Unsupported arch_type: {arch_type}')

        self.recursions = recursions
        self.arch_type = arch_type
        self.use_recursive_supervision = use_recursive_supervision

        # Embedding
        self.embedding_net = nn.Sequential(
            ConvAct(in_channels, hid_channels, 3, act_type=act_type),
            ConvAct(hid_channels, hid_channels, 3, act_type=act_type)
        )

        # Inference (recursive block) -- weight-shared
        self.inference_net = ConvAct(hid_channels, hid_channels, 3, act_type=act_type)

        # Reconstruction: map features back to image space (final conv no activation)
        self.reconstruction_net = nn.Sequential(
            ConvAct(hid_channels, hid_channels, 3, act_type=act_type),
            nn.Conv2d(hid_channels, out_channels, kernel_size=3, padding=1, bias=True)
        )

        # learnable fusion weights for side outputs (one weight per recursion)
        # initialize to equal logits so softmax -> uniform
        if self.use_recursive_supervision:
            init_logits = torch.ones(self.recursions, dtype=torch.float32) / float(self.recursions)
            # store as logits (un-normalized), learnable
            self.side_weight_logits = nn.Parameter(init_logits.clone(), requires_grad=True)

    def forward(self, x, return_all=False):
        """
        Args:
            x: input tensor [N, C, H, W]
            return_all (bool): if True and use_recursive_supervision==True, returns (fused, side_outputs_list, fusion_weights)
                               else returns (fused, 0, 0) for backward compatibility.
        """
        if self.arch_type == 'advanced':
            skip = x

        feat = self.embedding_net(x)

        side_outputs = []  # collect side outputs for supervision/fusion
        if self.arch_type == 'advanced':
            # advanced: every recursion produce a side-output using feat + skip
            for i in range(self.recursions):
                feat = self.inference_net(feat)
                out_i = self.reconstruction_net(feat + skip)
                if self.use_recursive_supervision:
                    side_outputs.append(out_i)
                else:
                    # if not collecting side outputs, we can still accumulate (like earlier simplified version)
                    if i == 0:
                        res = out_i
                    else:
                        res = res + out_i
        else:
            # basic: do recursions then single reconstruction; for compatibility we still optionally produce identical-side outputs
            for i in range(self.recursions):
                feat = self.inference_net(feat)
                if self.use_recursive_supervision:
                    out_i = self.reconstruction_net(feat)  # no skip
                    side_outputs.append(out_i)
            if not self.use_recursive_supervision:
                res = self.reconstruction_net(feat)

        # If using recursive-supervision: fuse side_outputs with learnable weights
        if self.use_recursive_supervision:
            # side_outputs: list of tensors [recursions x (N,C,H,W)]
            # normalize logits with softmax to get fusion weights
            weights = torch.softmax(self.side_weight_logits, dim=0)  # shape: (recursions,)
            # fuse: weighted sum over side outputs
            fused = None
            for i, out_i in enumerate(side_outputs):
                w = weights[i]
                # ensure shape/broadcast: multiply scalar weight
                if fused is None:
                    fused = w * out_i
                else:
                    fused = fused + w * out_i
            # fused is the final output (same shape as each side output)
            # For convenience, also provide side_outputs as a tensor of shape (R, N, C, H, W)
            side_stack = torch.stack(side_outputs, dim=0)  # (R, N, C, H, W)
            if return_all:
                # return fused output, side outputs list/tensor, and fusion weights (as 1D tensor)
                return fused, side_stack, weights
            else:
                return fused, 0, 0
        else:
            # Not using recursive supervision (backward-compatible path)
            return res, 0, 0
