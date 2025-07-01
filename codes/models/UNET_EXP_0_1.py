import torch
import torch.nn as nn
import torch.nn.functional as F

class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNET, self).__init__()

        # 下采样部分
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # 上采样部分
        self.decoder4 = self.upconv_block(1024, 512)
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = nn.Conv2d(128, out_channels, kernel_size=1)  # 输出通道调整为out_channels

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 下采样
        enc1 = self.encoder1(x)                     # (B, 64, 64, 64)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2)) # (B, 128, 32, 32)
        enc3 = self.encoder3(F.max_pool2d(enc2, 2)) # (B, 256, 16, 16)
        enc4 = self.encoder4(F.max_pool2d(enc3, 2)) # (B, 512, 8, 8)

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2)) # (B, 1024, 4, 4)

        # 上采样
        dec4 = self.decoder4(bottleneck)               # (B, 512, 8, 8)
        dec4 = torch.cat((dec4, enc4), dim=1)          # 跳跃连接
        dec3 = self.decoder3(dec4)                      # (B, 256, 16, 16)
        dec3 = torch.cat((dec3, enc3), dim=1)          # 跳跃连接
        dec2 = self.decoder2(dec3)                      # (B, 128, 32, 32)
        dec2 = torch.cat((dec2, enc2), dim=1)          # 跳跃连接
        dec1 = self.decoder1(dec2)                      # (B, out_channels, 64, 64)

        return dec1,0,0
