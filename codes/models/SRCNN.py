"""
Paper:      Image Super-Resolution Using Deep Convolutional Networks
Url:        https://arxiv.org/abs/1501.00092
Create by:  zh320
Date:       2023/12/09
"""

import torch.nn as nn
import torch.nn.functional as F

class Activation(nn.Module):
    def __init__(self, act_type, **kwargs):
        super(Activation, self).__init__()
        activation_hub = {'relu': nn.ReLU,             'relu6': nn.ReLU6,
                          'leakyrelu': nn.LeakyReLU,    'prelu': nn.PReLU,
                          'celu': nn.CELU,              'elu': nn.ELU, 
                          'hardswish': nn.Hardswish,    'hardtanh': nn.Hardtanh,
                          'gelu': nn.GELU,              'glu': nn.GLU, 
                          'selu': nn.SELU,              'silu': nn.SiLU,
                          'sigmoid': nn.Sigmoid,        'softmax': nn.Softmax, 
                          'tanh': nn.Tanh,              'none': nn.Identity,
                        }
                        
        act_type = act_type.lower()
        if act_type not in activation_hub.keys():
            raise NotImplementedError(f'Unsupport activation type: {act_type}')
        
        self.activation = activation_hub[act_type](**kwargs)
        
    def forward(self, x):
        return self.activation(x)
    
class ConvAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, 
                    groups=1, bias=True, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):    
            padding = (kernel_size - 1) // 2 * dilation
            
        super(ConvAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            Activation(act_type, **kwargs)
        )

def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, 
                    padding=2, bias=True)


class SRCNN(nn.Module):
    def __init__(self, in_channels, out_channels, upscale, kernel_setting='935', 
                    act_type='relu'):
        super(SRCNN, self).__init__()
        if kernel_setting not in ['915', '935', '955']:
            raise ValueError(f'Unknown kernel setting: {kernel_setting}. You can choose \
                                from ["915", "935", "955"].\n')
        kernel_map = {'915':1, '935':3, '955':5}

        self.upscale = upscale
        self.layer1 = ConvAct(in_channels, 64, 9, act_type=act_type)
        self.layer2 = ConvAct(64, 32, kernel_map[kernel_setting], act_type=act_type)
        self.layer3 = conv5x5(32, out_channels)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.upscale, mode='bicubic')
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x