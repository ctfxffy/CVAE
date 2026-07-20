import pytest
import torch

from src.models.cvae import CVAE
from src.training.utils import (
    denormalize,
    load_checkpoint,
    save_checkpoint,
    save_image_grid,
)


def test_checkpoint_roundtrip(tmp_path):
    model = CVAE(latent_dim=16, attr_dim=8, image_size=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint({"model": model.state_dict(), "optimizer": opt.state_dict(),
                     "epoch": 3, "best_val": 0.5}, path)
    model2 = CVAE(latent_dim=16, attr_dim=8, image_size=64)
    meta = load_checkpoint(path, model2, device="cpu")
    assert meta["epoch"] == 3
    assert meta["best_val"] == 0.5
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_load_checkpoint_shape_mismatch(tmp_path):
    model = CVAE(latent_dim=16, attr_dim=8, image_size=64)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint({"model": model.state_dict(), "epoch": 1, "best_val": 1.0}, path)
    wrong = CVAE(latent_dim=32, attr_dim=8, image_size=64)
    with pytest.raises(SystemExit):
        load_checkpoint(path, wrong, device="cpu")


def test_denormalize_range():
    imgs = torch.rand(2, 3, 8, 8) * 2 - 1
    out = denormalize(imgs)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_save_image_grid_creates_file(tmp_path):
    imgs = torch.rand(8, 3, 64, 64) * 2 - 1
    path = str(tmp_path / "grid.png")
    save_image_grid(imgs, path, nrow=4)
    assert (tmp_path / "grid.png").exists()
