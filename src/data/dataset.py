import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

_SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def read_split_file(path: str) -> dict:
    """读取 list_eval_partition.txt，返回 文件名 -> 0/1/2。"""
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                result[parts[0]] = int(parts[1])
    return result


def read_attr_file(path: str, attr_names: list) -> dict:
    """读取 list_attr_celeba.txt，返回 文件名 -> 0/1 属性列表（按 attr_names 顺序）。"""
    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # 第一行是图片总数
        header = f.readline().split()  # 第二行是 40 个属性名
        col_idx = [header.index(name) for name in attr_names]
        result = {}
        for line in f:
            parts = line.split()
            if len(parts) != 41:
                continue
            vals = parts[1:]
            result[parts[0]] = [1 if int(vals[j]) == 1 else 0 for j in col_idx]
    return result


class CelebAAttrDataset(Dataset):
    """CelebA 图片 + 8 维 0/1 属性向量。"""

    def __init__(self, root: str, attr_names: list, split: str = "train", image_size: int = 64):
        self.root = Path(root)
        code = _SPLIT_CODE[split]
        split_map = read_split_file(str(self.root / "list_eval_partition.txt"))
        attr_map = read_attr_file(str(self.root / "list_attr_celeba.txt"), attr_names)
        self.files = [n for n in sorted(split_map) if split_map[n] == code and n in attr_map]
        if not self.files:
            raise FileNotFoundError(f"{root} 下没有 split={split} 的样本，请先运行下载脚本")
        self.attr_map = attr_map
        self.transform = transforms.Compose([
            transforms.CenterCrop(148),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 坏图跳过：最多连续尝试 10 次
        for _ in range(10):
            name = self.files[idx]
            try:
                img = Image.open(self.root / "img_align_celeba" / name).convert("RGB")
                img = self.transform(img)
                attr = torch.tensor(self.attr_map[name], dtype=torch.float32)
                return img, attr
            except Exception:
                logger.warning("跳过损坏的图片: %s", name)
                idx = (idx + 1) % len(self.files)
        raise RuntimeError("连续 10 张图片损坏，数据集可能不完整")


def build_dataloaders(cfg: dict):
    train_ds = CelebAAttrDataset(cfg["data_root"], cfg["attr_names"], "train", cfg["image_size"])
    val_ds = CelebAAttrDataset(cfg["data_root"], cfg["attr_names"], "val", cfg["image_size"])
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    return train_loader, val_loader
