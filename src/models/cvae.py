import torch
import torch.nn.functional as F
from torch import nn

from src.models.decoder import Decoder
from src.models.encoder import Encoder


class CVAE(nn.Module):
    """条件变分自编码器：编码器/解码器组合 + 重参数化 + 采样。"""

    def __init__(self, latent_dim: int = 256, attr_dim: int = 8, image_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, attr_dim, image_size)
        self.decoder = Decoder(latent_dim, attr_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, attr):
        mu, logvar = self.encoder(x, attr)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, attr), mu, logvar

    @torch.no_grad()
    def sample(self, attr, z=None):
        if z is None:
            z = torch.randn(attr.size(0), self.latent_dim, device=attr.device)
        return self.decoder(z.to(attr.device), attr)


def cvae_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    """L = MSE + β·KL，均按 batch 取均值。"""
    recon = F.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total = recon + beta * kl
    return total, recon, kl
