import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.utils import save_image


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, device: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
    except RuntimeError as e:
        raise SystemExit(
            f"checkpoint 与当前模型结构不匹配（可能改过 latent_dim 等参数）：{path}\n{e}"
        ) from e
    return ckpt


def denormalize(imgs):
    """[-1,1] → [0,1]"""
    return (imgs + 1) / 2


def save_image_grid(images, path: str, nrow: int = 8) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(images.clamp(-1, 1)).cpu(), path, nrow=nrow)
