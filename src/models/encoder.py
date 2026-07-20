import torch
from torch import nn


class Encoder(nn.Module):
    """image + attr → μ, logσ²。属性摊平成 1 通道特征图与图像拼接。"""

    def __init__(self, latent_dim: int = 256, attr_dim: int = 8, image_size: int = 64):
        super().__init__()
        self.image_size = image_size
        self.attr_fc = nn.Linear(attr_dim, image_size * image_size)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
        )
        flat_dim = 512 * 4 * 4
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x, attr):
        b = x.size(0)
        a = self.attr_fc(attr).view(b, 1, self.image_size, self.image_size)
        h = torch.cat([x, a], dim=1)
        h = self.conv(h).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)
