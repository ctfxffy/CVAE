import torch
from torch.utils.data import DataLoader

from src.data.dataset import CelebAAttrDataset
from src.evaluate import evaluate_recon
from src.models.cvae import CVAE

ATTRS = ["Male", "Smiling", "Eyeglasses", "Mustache",
         "Blond_Hair", "Black_Hair", "Young", "Wearing_Lipstick"]


def test_evaluate_recon_returns_finite_floats(fake_celeba_root):
    ds = CelebAAttrDataset(fake_celeba_root, ATTRS, "val", 64)
    loader = DataLoader(ds, batch_size=2)
    model = CVAE(latent_dim=16, attr_dim=8, image_size=64)
    mse, kl = evaluate_recon(model, loader, torch.device("cpu"), num_batches=1)
    assert mse >= 0 and kl >= 0
    assert mse == mse and kl == kl  # 非 NaN
