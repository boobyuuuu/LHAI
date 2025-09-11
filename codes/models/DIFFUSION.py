from tqdm import tqdm
import torch
import torch.nn as nn
import deeplay as dl

class Diffusion:
    """Denoising diffusion probabilstic model (DDPM)."""

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02,
                img_size=64, device=None):
        """Initialize the diffusion model."""
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = (1.0 - self.beta).to(self.device)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(self.device)
        self.img_size = img_size

    def prepare_noise_schedule(self):
        """Prepare the noise schedule."""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def forward_diffusion(self, x0, t):
        """Implement the forward diffusion process."""
        device = t.device
        alpha_bar = self.alpha_bar.to(device)  # ✅ 将 alpha_bar 移到和 t 一样的 device

        sqrt_alpha_bar = torch.sqrt(alpha_bar[t])[:, None, None, None]  # ✅ 正确索引 alpha_bar
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t])[:, None, None, None]  # ✅ 同样修复

        noise = torch.randn_like(x0)
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

        return x_t, noise

    def reverse_diffusion(self, model, n_images, n_channels,
                      position_encoding_dim, position_encoding_function,
                      save_time_steps=None, input_image=None):
        """Reverse diffusion process"""
        with torch.no_grad():
            device = self.device
    
            x = torch.randn((n_images, n_channels, self.img_size, self.img_size)).to(device)
            input_image = input_image.to(device)
            model = model.to(device)
    
            # 确保 alpha 和 alpha_bar 在正确设备
            alpha = self.alpha.to(device)
            alpha_bar = self.alpha_bar.to(device)
    
            denoised_images = []
            for i in tqdm(reversed(range(0, self.noise_steps)), desc="U-Net inference", total=self.noise_steps):
                t = torch.full((n_images,), i, dtype=torch.long, device=device)
    
                t_pos_enc = position_encoding_function(t.unsqueeze(1), position_encoding_dim).to(device)
    
                xt_input = torch.cat((input_image, x), dim=1).to(device)  # ← 确保在 GPU
    
                predicted_noise = model(xt_input, t_pos_enc)  # 输入 now 确保都在 CUDA 上
    
                alpha_t = alpha[t][:, None, None, None]
                alpha_bar_t = alpha_bar[t][:, None, None, None]
    
                noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
    
                x = (
                    (1 / torch.sqrt(alpha_t)) * (
                        x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
                    ) + torch.sqrt(1 - alpha_t) * noise
                )
    
                if save_time_steps and i in save_time_steps:
                    denoised_images.append(x)
    
            denoised_images = torch.stack(denoised_images)
            denoised_images = denoised_images.swapaxes(0, 1)
            return denoised_images


def positional_encoding(t, enc_dim):
    """Encode position informaiton with a sinusoid."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, enc_dim, 2).float()
    / enc_dim)).to(t.device)
    pos_enc_a = torch.sin(t.repeat(1, enc_dim // 2) * inv_freq)
    pos_enc_b = torch.cos(t.repeat(1, enc_dim // 2) * inv_freq)
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class EnhancedUNetWrapper(nn.Module):
    def __init__(self, base_unet):
        super().__init__()
        self.resblock_in = ResidualBlock(2)     # 对输入做初步处理（可选）
        self.unet = base_unet
        self.resblock_out = ResidualBlock(1)    # 对输出做细节增强（可选）

    def forward(self, x, t=None):  # 有些 UNet 需要 timestep t
        x = self.resblock_in(x)
        if t is not None:
            out = self.unet(x, t)
        else:
            out = self.unet(x)
        out = self.resblock_out(out)
        return out
    
def prepare_data(input_image, target_image, diffusion, noise_steps, device, position_encoding_dim):
    """Prepare data."""

    batch_size = input_image.shape[0]

    input_image = input_image.to(device)
    target_image = target_image.to(device)

    t = torch.randint(low=0, high=noise_steps, size=(batch_size,)).to(device)

    x_t, noise = diffusion.forward_diffusion(target_image, t)
    x_t = torch.cat((input_image, x_t), dim=1)

    t = positional_encoding(t.unsqueeze(1), position_encoding_dim)

    return x_t.to(device), t.to(device), noise.to(device)