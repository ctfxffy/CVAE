import argparse
import logging

import torch

from src.data.dataset import build_dataloaders
from src.models.cvae import CVAE
from src.training.trainer import Trainer
from src.training.utils import load_config, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="训练 CVAE（CelebA 属性条件人脸生成）")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--resume", default=None, help="续训 checkpoint 路径，如 checkpoints/latest.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("使用设备: %s", device)

    train_loader, val_loader = build_dataloaders(cfg)
    model = CVAE(cfg["latent_dim"], len(cfg["attr_names"]), cfg["image_size"])
    logging.info("模型参数量: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    trainer = Trainer(model, train_loader, val_loader, cfg, device)
    if args.resume:
        trainer.resume(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
