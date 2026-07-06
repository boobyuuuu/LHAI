import torch
import torch.nn as nn
from tqdm import tqdm


class DDPM:
    """
    极简 DDPM（功能覆盖原 DIFFUSION）：
    - 内置正弦时间步嵌入（Transformer 同款）
    - 条件扩散（拼接条件图像）
    - 训练数据准备、反向采样（可保存中间快照）
    """

    def __init__(
        self,
        noise_steps: int = 1000,
        beta_start: float = 1e-6,
        beta_end: float = 0.01,
        img_size: int = 64,
        device: torch.device | str | None = None,
        pos_emb_dim: int = 256,
        conditional: bool = True,
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conditional = conditional

        # 噪声调度
        self.beta = torch.linspace(beta_start, beta_end, noise_steps, device=self.device)
        self.alpha = (1.0 - self.beta)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # 时间步嵌入
        self.pos_emb_dim = pos_emb_dim

    # ---------- utils ----------
    def _positional_encoding(self, t: torch.Tensor) -> torch.Tensor:
        """
        正弦时间步编码：t (B,) or (B,1) -> (B, D)
        """
        if t.ndim == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        device = t.device
        half = self.pos_emb_dim // 2
        # 防除零
        denom = max(1, half - 1)
        freq = torch.exp(
            torch.arange(0, half, device=device, dtype=torch.float32)
            * -(torch.log(torch.tensor(10000.0)) / denom)
        )
        ang = t.float().unsqueeze(1) * freq.unsqueeze(0)  # (B, half)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B, D)

    # ---------- q(x_t | x_0) ----------
    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor):
        """
        前向加噪：q(x_t | x_0)
        x0: (B, 1, H, W), t: (B,)
        """
        alpha_bar = self.alpha_bar.to(t.device)
        sqrt_ab = torch.sqrt(alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_ab = torch.sqrt(1 - alpha_bar[t])[:, None, None, None]
        noise = torch.randn_like(x0)
        x_t = sqrt_ab * x0 + sqrt_one_minus_ab * noise
        return x_t, noise

    # ---------- training helper ----------
    def sample_training_batch(self, input_img: torch.Tensor, target_img: torch.Tensor):
        """
        训练数据准备：
          返回 (x_t_concat, t_emb, noise)
          - 若 conditional=True，则 x_t_concat = cat(input_img, x_t)
        """
        B = target_img.size(0)
        device = self.device
        input_img = input_img.to(device)
        target_img = target_img.to(device)

        t = torch.randint(low=0, high=self.noise_steps, size=(B,), device=device)
        x_t, noise = self.forward_diffusion(target_img, t)

        if self.conditional:
            x_t = torch.cat([input_img, x_t], dim=1)  # (B, 2, H, W)

        t_emb = self._positional_encoding(t)  # (B, D)
        return x_t, t_emb, noise

    # ---------- p(x_{t-1} | x_t) ----------
    @torch.no_grad()
    def reverse_diffusion(
        self,
        model: nn.Module,
        n_images: int,
        n_channels: int,
        input_image: torch.Tensor | None = None,
        save_time_steps: list[int] | None = None,
        return_intermediates: bool = False,
    ):
        """
        反向采样：
          - 若 return_intermediates=True 且提供 save_time_steps，将返回中间快照 (B, K, C, H, W)
          - 否则返回最终图像 x_0，shape = (B, C, H, W)
        """
        device = self.device
        x = torch.randn((n_images, n_channels, self.img_size, self.img_size), device=device)

        if self.conditional and input_image is None:
            raise ValueError("conditional=True 时需要提供 input_image。")
        if input_image is not None:
            input_image = input_image.to(device)

        alpha = self.alpha.to(device)
        alpha_bar = self.alpha_bar.to(device)

        snaps = []
        for i in tqdm(reversed(range(0, self.noise_steps)), desc="Sampling", total=self.noise_steps):
            t = torch.full((n_images,), i, dtype=torch.long, device=device)
            t_emb = self._positional_encoding(t)  # (B, D)

            xt_input = x
            if self.conditional and input_image is not None:
                xt_input = torch.cat([input_image, x], dim=1)

            predicted_noise = model(xt_input, t_emb)  # 噪声预测

            alpha_t = alpha[t][:, None, None, None]
            alpha_bar_t = alpha_bar[t][:, None, None, None]
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)

            x = (1.0 / torch.sqrt(alpha_t)) * (
                x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise
            ) + torch.sqrt(1.0 - alpha_t) * noise

            if return_intermediates and save_time_steps and i in save_time_steps:
                snaps.append(x)

        if return_intermediates and save_time_steps:
            # (K, B, C, H, W) -> (B, K, C, H, W)
            return torch.stack(snaps, dim=0).swapaxes(0, 1)
        return x  # 最终 x_0
