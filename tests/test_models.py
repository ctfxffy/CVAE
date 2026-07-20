import torch

from src.models.decoder import Decoder
from src.models.encoder import Encoder


def test_encoder_output_shapes():
    enc = Encoder(latent_dim=256, attr_dim=8, image_size=64)
    x = torch.randn(4, 3, 64, 64)
    attr = torch.rand(4, 8)
    mu, logvar = enc(x, attr)
    assert mu.shape == (4, 256)
    assert logvar.shape == (4, 256)


def test_encoder_uses_attr():
    enc = Encoder(latent_dim=256, attr_dim=8, image_size=64)
    x = torch.randn(2, 3, 64, 64)
    mu_a, _ = enc(x, torch.zeros(2, 8))
    mu_b, _ = enc(x, torch.ones(2, 8))
    assert not torch.allclose(mu_a, mu_b)


def test_decoder_output_shape_and_range():
    dec = Decoder(latent_dim=256, attr_dim=8)
    z = torch.randn(4, 256)
    attr = torch.rand(4, 8)
    out = dec(z, attr)
    assert out.shape == (4, 3, 64, 64)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_decoder_uses_attr():
    dec = Decoder(latent_dim=256, attr_dim=8)
    z = torch.randn(2, 256)
    out_a = dec(z, torch.zeros(2, 8))
    out_b = dec(z, torch.ones(2, 8))
    assert not torch.allclose(out_a, out_b)
