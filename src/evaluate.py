import argparse

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CelebAAttrDataset
from src.models.cvae import CVAE, cvae_loss
from src.training.utils import load_checkpoint, load_config, save_image_grid


@torch.no_grad()
def evaluate_recon(model, loader, device, num_batches: int):
    """在 val 前 num_batches 个 batch 上计算平均 MSE 与 KL。"""
    model.eval()
    mse_sum, kl_sum, n_batches = 0.0, 0.0, 0
    for i, (x, attr) in enumerate(loader):
        if i >= num_batches:
            break
        x, attr = x.to(device), attr.to(device)
        recon, mu, logvar = model(x, attr)
        _, mse, kl = cvae_loss(recon, x, mu, logvar)
        mse_sum += mse.item()
        kl_sum += kl.item()
        n_batches += 1
    n = max(n_batches, 1)
    return mse_sum / n, kl_sum / n


@torch.no_grad()
def save_recon_grid(model, loader, device, out_path, n: int = 8):
    """上排原图、下排重建，保存对比网格。"""
    model.eval()
    x, attr = next(iter(loader))
    x, attr = x[:n].to(device), attr[:n].to(device)
    recon, _, _ = model(x, attr)
    save_image_grid(torch.cat([x, recon], dim=0), out_path, nrow=n)


def main():
    parser = argparse.ArgumentParser(description="评估 CVAE：重建质量 + 重建对比图")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--num-batches", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(cfg["latent_dim"], len(cfg["attr_names"]), cfg["image_size"])
    load_checkpoint(args.checkpoint, model, str(device))
    model.to(device)

    ds = CelebAAttrDataset(cfg["data_root"], cfg["attr_names"], "val", cfg["image_size"])
    loader = DataLoader(ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])

    mse, kl = evaluate_recon(model, loader, device, args.num_batches)
    print(f"val 前 {args.num_batches} 个 batch: MSE={mse:.4f}  KL={kl:.4f}")

    out = "outputs/eval/recon_grid.png"
    save_recon_grid(model, loader, device, out)
    print(f"重建对比图已保存: {out}")


if __name__ == "__main__":
    main()
