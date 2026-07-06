import torch
import torch.nn as nn
from tqdm import tqdm

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Standard sinusoidal positional embeddings, same as in Transformer.
    Input: timestep tensor (B,) or (B,1)
    Output: (B, dim)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or (B,1) int64 tensor, diffusion timesteps
        Returns:
            pos_enc: (B, dim) tensor
        """
        device = t.device
        half_dim = self.dim // 2
        emb_factor = torch.exp(
            torch.arange(0, half_dim, device=device, dtype=torch.float32)
            * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        # shape: (B, half_dim)
        emb = t.float().unsqueeze(1) * emb_factor.unsqueeze(0)
        pos_enc = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return pos_enc  # (B, dim)

class DDPM_Transformer:
    """Denoising Diffusion Probabilistic Model (DDPM_Transformer)."""

    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=64,
        device=None,
        pos_emb_dim=128,
        conditional=True,
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conditional = conditional

        # Noise schedule
        self.beta = torch.linspace(beta_start, beta_end, noise_steps, device=self.device)
        self.alpha = (1.0 - self.beta).to(self.device)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(self.device)

        # Positional encoding (sinusoidal, fixed)
        self.pos_emb_dim = pos_emb_dim
        self.pos_emb = SinusoidalPositionEmbeddings(pos_emb_dim).to(self.device)

    # -------------------
    # Forward diffusion (q)
    # -------------------
    def forward_diffusion(self, x0, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]
        noise = torch.randn_like(x0)
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    # -------------------
    # Prepare batch for training
    # -------------------
    def sample_training_batch(self, input_img, target_img):
        b = target_img.shape[0]
        t = torch.randint(low=0, high=self.noise_steps, size=(b,), device=self.device)

        x_t, noise = self.forward_diffusion(target_img, t)

        if self.conditional:
            x_t = torch.cat((input_img.to(self.device), x_t), dim=1)

        t_emb = self.pos_emb(t)  # sinusoidal encoding

        return x_t.to(self.device), t_emb.to(self.device), noise.to(self.device)

    # -------------------
    # Reverse diffusion (p)
    # -------------------
    @torch.no_grad()
    def reverse_diffusion(self, model, n_images, n_channels, input_image=None, save_time_steps=None):
        x = torch.randn((n_images, n_channels, self.img_size, self.img_size), device=self.device)
        alpha = self.alpha
        alpha_bar = self.alpha_bar

        denoised_images = []
        for i in tqdm(reversed(range(0, self.noise_steps)), desc="Sampling", total=self.noise_steps):
            t = torch.full((n_images,), i, dtype=torch.long, device=self.device)

            t_emb = self.pos_emb(t)

            xt_input = x
            if self.conditional and input_image is not None:
                xt_input = torch.cat((input_image.to(self.device), x), dim=1)

            predicted_noise = model(xt_input, t_emb)

            alpha_t = alpha[t][:, None, None, None]
            alpha_bar_t = alpha_bar[t][:, None, None, None]
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
            ) + torch.sqrt(1 - alpha_t) * noise

            if save_time_steps and i in save_time_steps:
                denoised_images.append(x)

        if save_time_steps:
            denoised_images = torch.stack(denoised_images).swapaxes(0, 1)
            return denoised_images
        return x
