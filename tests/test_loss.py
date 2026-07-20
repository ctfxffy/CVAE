import torch

from src.models.cvae import CVAE, cvae_loss


def test_cvae_forward_shapes():
    model = CVAE(latent_dim=256, attr_dim=8, image_size=64)
    x = torch.rand(4, 3, 64, 64) * 2 - 1
    attr = torch.rand(4, 8)
    recon, mu, logvar = model(x, attr)
    assert recon.shape == x.shape
    assert mu.shape == (4, 256)
    assert logvar.shape == (4, 256)


def test_reparameterize_gradient_flows():
    model = CVAE(latent_dim=256, attr_dim=8, image_size=64)
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    attr = torch.rand(2, 8)
    recon, mu, logvar = model(x, attr)
    loss, _, _ = cvae_loss(recon, x, mu, logvar, beta=1.0)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_loss_positive_and_finite():
    recon_x = torch.rand(4, 3, 64, 64) * 2 - 1
    x = torch.rand(4, 3, 64, 64) * 2 - 1
    mu = torch.randn(4, 256)
    logvar = torch.randn(4, 256)
    total, recon, kl = cvae_loss(recon_x, x, mu, logvar, beta=1.0)
    for v in (total, recon, kl):
        assert torch.isfinite(v)
    assert total.item() > 0


def test_overfit_single_batch_loss_decreases():
    torch.manual_seed(0)
    model = CVAE(latent_dim=32, attr_dim=8, image_size=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(8, 3, 64, 64) * 2 - 1
    attr = torch.rand(8, 8)
    recon, mu, logvar = model(x, attr)
    first, _, _ = cvae_loss(recon, x, mu, logvar, beta=0.1)
    for _ in range(50):
        opt.zero_grad()
        recon, mu, logvar = model(x, attr)
        loss, _, _ = cvae_loss(recon, x, mu, logvar, beta=0.1)
        loss.backward()
        opt.step()
    assert loss.item() < first.item()


def test_sample_shape_without_z():
    model = CVAE(latent_dim=256, attr_dim=8, image_size=64)
    attr = torch.rand(3, 8)
    out = model.sample(attr)
    assert out.shape == (3, 3, 64, 64)
