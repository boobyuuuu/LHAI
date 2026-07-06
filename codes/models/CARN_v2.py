import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + residual)

class CascadingBlock(nn.Module):
    def __init__(self, channels):
        super(CascadingBlock, self).__init__()
        self.rb1 = ResidualBlock(channels)
        self.rb2 = ResidualBlock(channels)
        self.rb3 = ResidualBlock(channels)
        self.conv = nn.Conv2d(channels*3, channels, 1)
    
    def forward(self, x):
        x1 = self.rb1(x)
        x2 = self.rb2(x1)
        x3 = self.rb3(x2)
        concat = torch.cat([x1, x2, x3], dim=1)
        out = self.conv(concat)
        return out

class CARN_v2(nn.Module):
    def __init__(self, jpt):
        super(CARN_v2, self).__init__()
        self.entry = nn.Conv2d(1, 64, 3, padding=1)
        self.cascading1 = CascadingBlock(64)
        self.cascading2 = CascadingBlock(64)
        self.cascading3 = CascadingBlock(64)
        self.exit = nn.Conv2d(64, 1, 3, padding=1)
    
    def forward(self, x):
        x = self.entry(x)
        x = self.cascading1(x)
        x = self.cascading2(x)
        x = self.cascading3(x)
        x = self.exit(x)
        return x, 0, 0