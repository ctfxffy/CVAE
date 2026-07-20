import torch
from torch.utils.data import DataLoader

from src.data.dataset import CelebAAttrDataset
from src.models.cvae import CVAE
from src.training.trainer import Trainer

ATTRS = ["Male", "Smiling", "Eyeglasses", "Mustache",
         "Blond_Hair", "Black_Hair", "Young", "Wearing_Lipstick"]


def test_trainer_one_epoch_smoke(fake_celeba_root, tmp_path):
    cfg = {
        "data_root": fake_celeba_root,
        "attr_names": ATTRS,
        "image_size": 64,
        "latent_dim": 16,
        "beta": 1.0,
        "lr": 1e-3,
        "epochs": 1,
        "sample_every": 2,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "output_dir": str(tmp_path / "outputs"),
    }
    train_ds = CelebAAttrDataset(fake_celeba_root, ATTRS, "train", 64)
    val_ds = CelebAAttrDataset(fake_celeba_root, ATTRS, "val", 64)
    train_loader = DataLoader(train_ds, batch_size=4)
    val_loader = DataLoader(val_ds, batch_size=4)
    model = CVAE(latent_dim=16, attr_dim=8, image_size=64)
    trainer = Trainer(model, train_loader, val_loader, cfg, torch.device("cpu"))
    trainer.train()
    assert (tmp_path / "checkpoints" / "latest.pt").exists()
    assert (tmp_path / "checkpoints" / "best.pt").exists()
    samples = list((tmp_path / "outputs" / "samples").glob("step_*.png"))
    assert len(samples) >= 1
