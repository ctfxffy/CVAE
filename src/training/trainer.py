import logging
from pathlib import Path

import torch

from src.models.cvae import cvae_loss
from src.training.utils import load_checkpoint, save_checkpoint, save_image_grid

logger = logging.getLogger(__name__)


class Trainer:
    """CVAE 训练循环：AMP 混合精度、定期采样、latest/best checkpoint。"""

    def __init__(self, model, train_loader, val_loader, cfg: dict, device: torch.device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.amp = device.type == "cuda"
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.start_epoch = 1
        self.best_val = float("inf")
        self.global_step = 0
        # 固定采样素材：9 组属性（全 0 + 8 个单属性）× 8 个固定 z
        attr_dim = len(cfg["attr_names"])
        self.fixed_z = torch.randn(8, cfg["latent_dim"], device=device)
        combos = torch.cat([torch.zeros(1, attr_dim), torch.eye(attr_dim)], dim=0)
        self.sample_attrs = combos.to(device)

    def resume(self, path: str) -> None:
        ckpt = load_checkpoint(path, self.model, self.optimizer, str(self.device))
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val = ckpt["best_val"]
        logger.info("从 %s 恢复，epoch 从 %d 开始", path, self.start_epoch)

    def train(self) -> None:
        try:
            for epoch in range(self.start_epoch, self.cfg["epochs"] + 1):
                train_loss = self._run_epoch(self.train_loader, train=True)
                val_loss = self._run_epoch(self.val_loader, train=False)
                logger.info(
                    "epoch %d/%d  train=%.4f  val=%.4f",
                    epoch, self.cfg["epochs"], train_loss, val_loss,
                )
                self._save(epoch, val_loss)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise SystemExit(
                    "CUDA 显存不足：请在 configs/default.yaml 将 batch_size 减半后重试"
                ) from e
            raise

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train() if train else self.model.eval()
        total, count = 0.0, 0
        for x, attr in loader:
            x = x.to(self.device, non_blocking=True)
            attr = attr.to(self.device, non_blocking=True)
            with torch.set_grad_enabled(train):
                with torch.amp.autocast("cuda", enabled=self.amp):
                    recon, mu, logvar = self.model(x, attr)
                    loss, _, _ = cvae_loss(recon, x, mu, logvar, self.cfg["beta"])
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.global_step += 1
                    if self.global_step % self.cfg["sample_every"] == 0:
                        self._sample_grid()
            total += loss.item() * x.size(0)
            count += x.size(0)
        return total / max(count, 1)

    @torch.no_grad()
    def _sample_grid(self) -> None:
        self.model.eval()
        n_z = self.fixed_z.size(0)
        attrs = self.sample_attrs.repeat_interleave(n_z, dim=0)
        z = self.fixed_z.repeat(self.sample_attrs.size(0), 1)
        imgs = self.model.sample(attrs, z)
        path = Path(self.cfg["output_dir"]) / "samples" / f"step_{self.global_step:06d}.png"
        save_image_grid(imgs, str(path), nrow=n_z)
        self.model.train()

    def _save(self, epoch: int, val_loss: float) -> None:
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_val": min(self.best_val, val_loss),
        }
        ckpt_dir = Path(self.cfg["checkpoint_dir"])
        save_checkpoint(state, str(ckpt_dir / "latest.pt"))
        if val_loss < self.best_val:
            self.best_val = val_loss
            save_checkpoint(state, str(ckpt_dir / "best.pt"))
