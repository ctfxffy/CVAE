import torch
from torch import nn


class Decoder(nn.Module):
    """z + attr → image。属性与 z 拼接后经全连接泡开成特征图。"""

    def __init__(self, latent_dim: int = 256, attr_dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(latent_dim + attr_dim, 512 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, z, attr):
        h = self.fc(torch.cat([z, attr], dim=1)).view(-1, 512, 4, 4)
        return self.deconv(h)
