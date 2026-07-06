import torch
import torch.nn as nn

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                     padding=1, bias=True)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, 
                     padding=0, bias=True)

class Activation(nn.Module):
    def __init__(self, act_type='relu'):
        super(Activation, self).__init__()
        activation_hub = {
            'relu': nn.ReLU,
            'prelu': nn.PReLU,
            'leakyrelu': nn.LeakyReLU,
            'gelu': nn.GELU,
            'none': nn.Identity,
        }
        act_type = act_type.lower()
        if act_type not in activation_hub:
            raise NotImplementedError(f'Unsupport activation type: {act_type}')
        self.activation = activation_hub[act_type]()
        
    def forward(self, x):
        return self.activation(x)

class ConvAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act_type='relu'):
        padding = (kernel_size - 1) // 2
        super(ConvAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
            Activation(act_type)
        )

class ResidualBlock(nn.Module):
    def __init__(self, channels, act_type='relu'):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            ConvAct(channels, channels, 3, act_type=act_type),
            conv3x3(channels, channels)
        )
        self.act = Activation(act_type)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x += residual
        return self.act(x)

class CascadingBlock(nn.Module):
    def __init__(self, block, channels, act_type='relu'):
        super(CascadingBlock, self).__init__()
        self.res1 = block(channels, act_type)
        self.conv1 = conv1x1(2*channels, channels)
        self.res2 = block(channels, act_type)
        self.conv2 = conv1x1(3*channels, channels)
        self.res3 = block(channels, act_type)
        self.conv3 = conv1x1(4*channels, channels)

    def forward(self, x):
        x0 = x
        x1 = self.res1(x)
        x_cat1 = torch.cat([x0, x1], dim=1)
        x1_out = self.conv1(x_cat1)

        x2 = self.res2(x1_out)
        x_cat2 = torch.cat([x0, x1, x2], dim=1)
        x2_out = self.conv2(x_cat2)

        x3 = self.res3(x2_out)
        x_cat3 = torch.cat([x0, x1, x2, x3], dim=1)
        x3_out = self.conv3(x_cat3)

        return x3_out

class CARN_v1(nn.Module):
    """
    Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network
    ECCV 2018, Moon et al.
    This implementation is simplified for scale=1 (no upsampling, input/output shape unchanged).
    """
    def __init__(self, jpt, in_channels=1, out_channels=1, hid_channels=64, act_type='relu'):
        super(CARN_v1, self).__init__()
        block = ResidualBlock
        self.conv1 = conv3x3(in_channels, hid_channels)
        self.cascading_block1 = CascadingBlock(block, hid_channels, act_type)
        self.conv2 = conv1x1(2*hid_channels, hid_channels)
        self.cascading_block2 = CascadingBlock(block, hid_channels, act_type)
        self.conv3 = conv1x1(3*hid_channels, hid_channels)
        self.cascading_block3 = CascadingBlock(block, hid_channels, act_type)
        self.conv4 = conv1x1(4*hid_channels, hid_channels)
        self.conv_last = conv3x3(hid_channels, out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x_cb1 = self.cascading_block1(x1)
        x = torch.cat([x1, x_cb1], dim=1)
        x = self.conv2(x)
        x_cb2 = self.cascading_block2(x)
        x = torch.cat([x1, x_cb1, x_cb2], dim=1)
        x = self.conv3(x)
        x_cb3 = self.cascading_block3(x)
        x = torch.cat([x1, x_cb1, x_cb2, x_cb3], dim=1)
        x = self.conv4(x)
        x = self.conv_last(x)
        return x, 0, 0