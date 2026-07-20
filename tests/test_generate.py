import pytest
import torch

from src.generate import generate, parse_attrs
from src.models.cvae import CVAE
from src.training.utils import save_checkpoint

ATTRS = ["Male", "Smiling", "Eyeglasses", "Mustache",
         "Blond_Hair", "Black_Hair", "Young", "Wearing_Lipstick"]


def test_parse_attrs_partial_defaults_zero():
    v = parse_attrs("Smiling=1,Eyeglasses=1", ATTRS)
    assert v.shape == (1, 8)
    assert v[0, 1].item() == 1.0 and v[0, 2].item() == 1.0
    assert v[0, 0].item() == 0.0


def test_parse_attrs_none_all_zero():
    v = parse_attrs(None, ATTRS)
    assert v.sum().item() == 0.0


def test_parse_attrs_unknown_name():
    with pytest.raises(ValueError, match="Smilig"):
        parse_attrs("Smilig=1", ATTRS)


def test_parse_attrs_bad_value():
    with pytest.raises(ValueError):
        parse_attrs("Smiling=2", ATTRS)


def test_generate_saves_grid(tmp_path):
    model = CVAE(latent_dim=16, attr_dim=8, image_size=64)
    ckpt = str(tmp_path / "ckpt.pt")
    save_checkpoint({"model": model.state_dict(), "epoch": 1, "best_val": 1.0}, ckpt)
    cfg = {"latent_dim": 16, "attr_names": ATTRS, "image_size": 64}
    out = str(tmp_path / "gen.png")
    result = generate("Smiling=1", 8, ckpt, out, cfg, seed=42)
    assert result == out
    assert (tmp_path / "gen.png").exists()
