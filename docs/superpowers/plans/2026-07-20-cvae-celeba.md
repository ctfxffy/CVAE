# CVAE-CelebA 属性条件人脸生成 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 PyTorch 在 CelebA 上训练卷积 CVAE，实现"输入人脸属性 → 生成对应人脸图片"的命令行工具链。

**Architecture:** 卷积编码器（image+attr → μ,logσ²）+ 转置卷积解码器（z+attr → image），latent_dim=256，64×64 分辨率；数据/模型/训练三层分离，三个薄 CLI 入口（train/generate/evaluate）。

**Tech Stack:** Python 3.10，PyTorch ≥2.1（CUDA 12.1），torchvision，PyYAML，Pillow，pytest，gdown。

## Global Constraints

- 设计 spec：`docs/superpowers/specs/2026-07-20-cvae-celeba-design.md`（所有决策以此为准）
- Python 3.10.2；GPU RTX 4060 Laptop 8GB；AMP 混合精度训练
- 条件属性 8 个（顺序固定）：`Male, Smiling, Eyeglasses, Mustache, Blond_Hair, Black_Hair, Young, Wearing_Lipstick`
- 图像预处理：CenterCrop(148) → Resize(64×64) → ToTensor → Normalize 到 [-1,1]；属性 {-1,1} 映射 {0,1}
- 数据布局：`data/celeba/img_align_celeba/*.jpg`、`data/celeba/list_attr_celeba.txt`、`data/celeba/list_eval_partition.txt`（0=train 1=val 2=test）
- 损失：`MSE + β·KL`，β 默认 1.0
- 所有命令在 Git Bash 中运行；pytest 用 `python -m pytest`
- `data/`、`checkpoints/`、`outputs/` 不入库（.gitignore）
- 注释与日志信息用中文，标识符用英文

---

### Task 1: 项目脚手架 + 配置加载

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `configs/default.yaml`
- Create: `src/__init__.py`、`src/data/__init__.py`、`src/models/__init__.py`、`src/training/__init__.py`、`tests/__init__.py`（均为空文件）
- Create: `src/training/utils.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path: str) -> dict` — 后续所有任务通过它读 yaml 配置；配置键见 `configs/default.yaml`

- [ ] **Step 1: 写失败测试**

`tests/test_config.py`:

```python
from src.training.utils import load_config


def test_load_config_returns_expected_keys():
    cfg = load_config("configs/default.yaml")
    assert cfg["latent_dim"] == 256
    assert cfg["image_size"] == 64
    assert cfg["batch_size"] == 128
    assert len(cfg["attr_names"]) == 8
    assert cfg["attr_names"][0] == "Male"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'src.training.utils'`

- [ ] **Step 3: 写脚手架文件**

`.gitignore`:

```
__pycache__/
*.pyc
.pytest_cache/
data/
checkpoints/
outputs/
```

`requirements.txt`（torch/torchvision 不列入，避免从 PyPI 默认源重装时覆盖 CUDA 版；由 README 指引单独安装）:

```
PyYAML>=6.0
Pillow>=10.0
pytest>=7.0
gdown>=4.7
numpy>=1.24
```

`configs/default.yaml`:

```yaml
seed: 42
data_root: data/celeba
image_size: 64
attr_names: [Male, Smiling, Eyeglasses, Mustache, Blond_Hair, Black_Hair, Young, Wearing_Lipstick]
latent_dim: 256
beta: 1.0
batch_size: 128
num_workers: 4
lr: 0.001
epochs: 30
sample_every: 500
checkpoint_dir: checkpoints
output_dir: outputs
```

`src/training/utils.py`:

```python
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
```

创建 5 个空 `__init__.py`。

- [ ] **Step 4: 安装依赖并运行测试**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install PyYAML Pillow pytest gdown numpy
python -c "import torch; print(torch.cuda.is_available())"
python -m pytest tests/test_config.py -v
```

Expected: `torch.cuda.is_available()` 打印 `True`；测试 PASS

- [ ] **Step 5: Commit**

```bash
git add .gitignore requirements.txt configs src tests
git commit -m "feat: project scaffolding with config loader"
```

---

### Task 2: CelebA 属性数据集

**Files:**
- Create: `src/data/dataset.py`
- Test: `tests/conftest.py`、`tests/test_dataset.py`

**Interfaces:**
- Consumes: `load_config(path) -> dict`（Task 1）
- Produces:
  - `read_split_file(path: str) -> dict[str, int]` — 文件名 → 0/1/2
  - `read_attr_file(path: str, attr_names: list[str]) -> dict[str, list[int]]` — 文件名 → 0/1 属性列表（按 attr_names 顺序）
  - `CelebAAttrDataset(root: str, attr_names: list[str], split: str, image_size: int)`，`__getitem__` 返回 `(img: FloatTensor[3,64,64] ∈ [-1,1], attr: FloatTensor[8] ∈ {0,1})`
  - `build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]` — (train, val)

- [ ] **Step 1: 写测试夹具和失败测试**

`tests/conftest.py`:

```python
import numpy as np
import pytest
from PIL import Image

# CelebA 官方 40 个属性的顺序
ALL_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young",
]


@pytest.fixture
def fake_celeba_root(tmp_path):
    """构造 8 张假图的迷你 CelebA：6 train + 2 val。"""
    root = tmp_path / "celeba"
    img_dir = root / "img_align_celeba"
    img_dir.mkdir(parents=True)
    rng = np.random.RandomState(0)
    names = []
    for i in range(8):
        name = f"{i + 1:06d}.jpg"
        arr = rng.randint(0, 255, (218, 178, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / name)
        names.append(name)

    with open(root / "list_attr_celeba.txt", "w") as f:
        f.write("8\n")
        f.write(" ".join(ALL_ATTRS) + "\n")
        for i, name in enumerate(names):
            # 确定性模式：第 j 个属性 = 1 当且仅当 (i + j) 为偶数
            vals = ["1" if (i + j) % 2 == 0 else "-1" for j in range(40)]
            f.write(f"{name} {' '.join(vals)}\n")

    with open(root / "list_eval_partition.txt", "w") as f:
        for i, name in enumerate(names):
            f.write(f"{name} {0 if i < 6 else 1}\n")

    return str(root)
```

`tests/test_dataset.py`:

```python
import torch

from src.data.dataset import CelebAAttrDataset, read_attr_file, read_split_file

ATTRS = ["Male", "Smiling", "Eyeglasses", "Mustache",
         "Blond_Hair", "Black_Hair", "Young", "Wearing_Lipstick"]


def test_read_split_file(fake_celeba_root):
    split = read_split_file(f"{fake_celeba_root}/list_eval_partition.txt")
    assert split["000001.jpg"] == 0
    assert split["000008.jpg"] == 1
    assert len(split) == 8


def test_read_attr_file_maps_to_zero_one(fake_celeba_root):
    attrs = read_attr_file(f"{fake_celeba_root}/list_attr_celeba.txt", ATTRS)
    vec = attrs["000001.jpg"]
    assert len(vec) == 8
    assert set(vec) <= {0, 1}
    # 第 1 张图 (i=0)：Male 是第 20 列 (j=20)，(0+20) 偶数 → 1
    assert vec[0] == 1
    # Smiling 是第 31 列 (j=31)，(0+31) 奇数 → 0
    assert vec[1] == 0


def test_dataset_len_by_split(fake_celeba_root):
    train = CelebAAttrDataset(fake_celeba_root, ATTRS, split="train", image_size=64)
    val = CelebAAttrDataset(fake_celeba_root, ATTRS, split="val", image_size=64)
    assert len(train) == 6
    assert len(val) == 2


def test_dataset_item_shapes_and_range(fake_celeba_root):
    ds = CelebAAttrDataset(fake_celeba_root, ATTRS, split="train", image_size=64)
    img, attr = ds[0]
    assert img.shape == (3, 64, 64)
    assert img.min() >= -1.0 and img.max() <= 1.0
    assert attr.shape == (8,)
    assert set(attr.tolist()) <= {0.0, 1.0}
    assert attr.dtype == torch.float32
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'src.data.dataset'`

- [ ] **Step 3: 实现 dataset.py**

`src/data/dataset.py`:

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: 4 个测试全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/dataset.py tests/conftest.py tests/test_dataset.py
git commit -m "feat: CelebA attribute dataset with fake-data tests"
```

---

### Task 3: 编码器与解码器

**Files:**
- Create: `src/models/encoder.py`
- Create: `src/models/decoder.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `Encoder(latent_dim: int, attr_dim: int, image_size: int)`；`forward(x: Tensor[B,3,H,W], attr: Tensor[B,attr_dim]) -> (mu: Tensor[B,latent_dim], logvar: Tensor[B,latent_dim])`
  - `Decoder(latent_dim: int, attr_dim: int)`；`forward(z: Tensor[B,latent_dim], attr: Tensor[B,attr_dim]) -> Tensor[B,3,64,64]`（值域 [-1,1]）

- [ ] **Step 1: 写失败测试**

`tests/test_models.py`:

```python
import torch

from src.models.decoder import Decoder
from src.models.encoder import Encoder


def test_encoder_output_shapes():
    enc = Encoder(latent_dim=256, attr_dim=8, image_size=64)
    x = torch.randn(4, 3, 64, 64)
    attr = torch.rand(4, 8)
    mu, logvar = enc(x, attr)
    assert mu.shape == (4, 256)
    assert logvar.shape == (4, 256)


def test_encoder_uses_attr():
    enc = Encoder(latent_dim=256, attr_dim=8, image_size=64)
    x = torch.randn(2, 3, 64, 64)
    mu_a, _ = enc(x, torch.zeros(2, 8))
    mu_b, _ = enc(x, torch.ones(2, 8))
    assert not torch.allclose(mu_a, mu_b)


def test_decoder_output_shape_and_range():
    dec = Decoder(latent_dim=256, attr_dim=8)
    z = torch.randn(4, 256)
    attr = torch.rand(4, 8)
    out = dec(z, attr)
    assert out.shape == (4, 3, 64, 64)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_decoder_uses_attr():
    dec = Decoder(latent_dim=256, attr_dim=8)
    z = torch.randn(2, 256)
    out_a = dec(z, torch.zeros(2, 8))
    out_b = dec(z, torch.ones(2, 8))
    assert not torch.allclose(out_a, out_b)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'src.models.encoder'`

- [ ] **Step 3: 实现 encoder.py 和 decoder.py**

`src/models/encoder.py`:

```python
import torch
from torch import nn


class Encoder(nn.Module):
    """image + attr → μ, logσ²。属性摊平成 1 通道特征图与图像拼接。"""

    def __init__(self, latent_dim: int = 256, attr_dim: int = 8, image_size: int = 64):
        super().__init__()
        self.image_size = image_size
        self.attr_fc = nn.Linear(attr_dim, image_size * image_size)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
        )
        flat_dim = 512 * 4 * 4
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x, attr):
        b = x.size(0)
        a = self.attr_fc(attr).view(b, 1, self.image_size, self.image_size)
        h = torch.cat([x, a], dim=1)
        h = self.conv(h).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)
```

`src/models/decoder.py`:

```python
import torch
from torch import nn


class Decoder(nn.Module):
    """z + attr → image。属性与 z 拼接后经全连接泡开成特征图。"""

    def __init__(self, latent_dim: int = 256, attr_dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(latent_dim + attr_dim, 512 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, z, attr):
        h = self.fc(torch.cat([z, attr], dim=1)).view(-1, 512, 4, 4)
        return self.deconv(h)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_models.py -v`
Expected: 4 个测试全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/encoder.py src/models/decoder.py tests/test_models.py
git commit -m "feat: convolutional encoder and decoder with attribute conditioning"
```

---

### Task 4: CVAE 组合 + 损失函数

**Files:**
- Create: `src/models/cvae.py`
- Test: `tests/test_loss.py`

**Interfaces:**
- Consumes: `Encoder`、`Decoder`（Task 3）
- Produces:
  - `CVAE(latent_dim: int, attr_dim: int, image_size: int)`，含属性 `latent_dim`
  - `CVAE.forward(x, attr) -> (recon, mu, logvar)`
  - `CVAE.sample(attr: Tensor[B,attr_dim], z: Tensor | None = None) -> Tensor[B,3,64,64]`（no_grad）
  - `cvae_loss(recon_x, x, mu, logvar, beta: float) -> (total, recon_loss, kl)`，均为标量 Tensor

- [ ] **Step 1: 写失败测试**

`tests/test_loss.py`:

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_loss.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'src.models.cvae'`

- [ ] **Step 3: 实现 cvae.py**

`src/models/cvae.py`:

```python
import torch
import torch.nn.functional as F
from torch import nn

from src.models.decoder import Decoder
from src.models.encoder import Encoder


class CVAE(nn.Module):
    """条件变分自编码器：编码器/解码器组合 + 重参数化 + 采样。"""

    def __init__(self, latent_dim: int = 256, attr_dim: int = 8, image_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, attr_dim, image_size)
        self.decoder = Decoder(latent_dim, attr_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, attr):
        mu, logvar = self.encoder(x, attr)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, attr), mu, logvar

    @torch.no_grad()
    def sample(self, attr, z=None):
        if z is None:
            z = torch.randn(attr.size(0), self.latent_dim, device=attr.device)
        return self.decoder(z.to(attr.device), attr)


def cvae_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    """L = MSE + β·KL，均按 batch 取均值。"""
    recon = F.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total = recon + beta * kl
    return total, recon, kl
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_loss.py -v`
Expected: 5 个测试全部 PASS（过拟合测试约 10-20 秒）

- [ ] **Step 5: Commit**

```bash
git add src/models/cvae.py tests/test_loss.py
git commit -m "feat: CVAE model with reparameterization and loss"
```

---

### Task 5: 训练工具函数（种子、checkpoint、图像保存）

**Files:**
- Modify: `src/training/utils.py`（追加函数）
- Test: `tests/test_utils.py`

**Interfaces:**
- Produces:
  - `set_seed(seed: int) -> None`
  - `save_checkpoint(state: dict, path: str) -> None`
  - `load_checkpoint(path: str, model: nn.Module, optimizer=None, device: str = "cpu") -> dict`（返回 checkpoint dict，含 `epoch`、`best_val`；shape 不匹配时抛带中文提示的 SystemExit）
  - `denormalize(imgs: Tensor) -> Tensor`（[-1,1] → [0,1]）
  - `save_image_grid(images: Tensor, path: str, nrow: int = 8) -> None`

- [ ] **Step 1: 写失败测试**

`tests/test_utils.py`:

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_utils.py -v`
Expected: FAIL，`ImportError: cannot import name 'denormalize'`

- [ ] **Step 3: 在 utils.py 追加实现**

在 `src/training/utils.py` 末尾追加：

```python
import random

import numpy as np
import torch
from pathlib import Path
from torchvision.utils import save_image


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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_utils.py -v`
Expected: 4 个测试全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/training/utils.py tests/test_utils.py
git commit -m "feat: training utils (seed, checkpoint, image grid)"
```

---

### Task 6: 训练器 Trainer

**Files:**
- Create: `src/training/trainer.py`
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `CVAE`、`cvae_loss`（Task 4）；utils 全部（Task 5）；`CelebAAttrDataset`（Task 2）
- Produces:
  - `Trainer(model, train_loader, val_loader, cfg: dict, device: torch.device)`
  - `Trainer.train() -> None` — 完整训练循环；写 `checkpoints/latest.pt`、`checkpoints/best.pt`、`outputs/samples/step_XXXXXX.png`
  - `Trainer.resume(path: str) -> None` — 恢复 epoch/optimizer/best_val
  - cfg 额外键（测试用）：`checkpoint_dir`、`output_dir`、`sample_every`、`epochs`

- [ ] **Step 1: 写失败测试（1 epoch 端到端冒烟）**

`tests/test_trainer.py`:

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_trainer.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'src.training.trainer'`

- [ ] **Step 3: 实现 trainer.py**

`src/training/trainer.py`:

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_trainer.py -v`
Expected: PASS（CPU 上约 30-60 秒）

- [ ] **Step 5: Commit**

```bash
git add src/training/trainer.py tests/test_trainer.py
git commit -m "feat: trainer with AMP, periodic sampling, checkpointing"
```

---

### Task 7: 训练入口 train.py

**Files:**
- Create: `src/train.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `load_config`、`build_dataloaders`、`CVAE`、`Trainer`、`set_seed`
- Produces: CLI `python -m src.train --config configs/default.yaml [--resume checkpoints/latest.pt]`

- [ ] **Step 1: 写失败测试**

`tests/test_cli.py`:

```python
import subprocess
import sys


def run_cli(module, *args):
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True, text=True, timeout=120,
    )


def test_train_help():
    r = run_cli("src.train", "--help")
    assert r.returncode == 0
    assert "--config" in r.stdout
    assert "--resume" in r.stdout
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL，returncode != 0（模块不存在）

- [ ] **Step 3: 实现 train.py**

`src/train.py`:

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_cli.py
git commit -m "feat: train CLI entry with resume support"
```

---

### Task 8: 生成入口 generate.py

**Files:**
- Create: `src/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `CVAE`（Task 4）、`load_checkpoint`、`save_image_grid`、`load_config`（Task 5/1）
- Produces:
  - `parse_attrs(s: str | None, attr_names: list[str]) -> torch.Tensor[1, len(attr_names)]` — 未指定的属性默认 0；非法名称/取值抛 ValueError（信息含合法清单）
  - `generate(attrs_str, n, checkpoint, out_path, config, seed) -> str`（返回保存路径）
  - CLI `python -m src.generate --attrs "Smiling=1,Eyeglasses=1" -n 8 --seed 42`

- [ ] **Step 1: 写失败测试**

`tests/test_generate.py`:

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_generate.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'src.generate'`

- [ ] **Step 3: 实现 generate.py**

`src/generate.py`:

```python
import argparse

import torch

from src.models.cvae import CVAE
from src.training.utils import load_checkpoint, load_config, save_image_grid


def parse_attrs(s, attr_names):
    """解析 "Smiling=1,Male=0" 为 [1, len(attr_names)] 的 0/1 张量，未指定的为 0。"""
    vec = torch.zeros(1, len(attr_names))
    if not s:
        return vec
    for pair in s.split(","):
        name, sep, val = pair.strip().partition("=")
        if name not in attr_names:
            raise ValueError(
                f"未知属性 '{name}'，合法属性: {', '.join(attr_names)}"
            )
        if not sep or val not in ("0", "1"):
            raise ValueError(f"属性 '{name}' 的取值必须是 0 或 1，收到 '{val}'")
        vec[0, attr_names.index(name)] = float(val)
    return vec


def generate(attrs_str, n, checkpoint, out_path, config, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(config["latent_dim"], len(config["attr_names"]), config["image_size"])
    load_checkpoint(checkpoint, model, str(device))
    model.to(device).eval()

    attr = parse_attrs(attrs_str, config["attr_names"]).to(device)
    attr = attr.repeat(n, 1)
    z = torch.randn(n, config["latent_dim"], device=device)
    imgs = model.sample(attr, z)
    save_image_grid(imgs, out_path, nrow=min(n, 8))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="按属性生成人脸")
    parser.add_argument("--attrs", default=None,
                        help='属性组合，如 "Smiling=1,Eyeglasses=1,Male=0"；不填则全 0')
    parser.add_argument("-n", type=int, default=8, help="生成数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default=None, help="输出路径，默认 outputs/generated/<attrs>.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = args.out
    if out is None:
        tag = (args.attrs or "random").replace("=", "").replace(",", "_")
        out = f"outputs/generated/{tag}_n{args.n}_seed{args.seed}.png"
    path = generate(args.attrs, args.n, args.checkpoint, out, cfg, args.seed)
    print(f"已保存: {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_generate.py -v`
Expected: 5 个测试全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/generate.py tests/test_generate.py
git commit -m "feat: generate CLI with attribute parsing"
```

---

### Task 9: 评估入口 evaluate.py

**Files:**
- Create: `src/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `CVAE`、`cvae_loss`、`CelebAAttrDataset`、`load_checkpoint`、`save_image_grid`、`load_config`
- Produces:
  - `evaluate_recon(model, loader, device, num_batches: int) -> tuple[float, float]` — (平均 MSE, 平均 KL)
  - `save_recon_grid(model, loader, device, out_path, n: int = 8) -> None` — 上排原图下排重建
  - CLI `python -m src.evaluate [--num-batches 20] [--checkpoint checkpoints/best.pt]`

- [ ] **Step 1: 写失败测试**

`tests/test_evaluate.py`:

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'src.evaluate'`

- [ ] **Step 3: 实现 evaluate.py**

`src/evaluate.py`:

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluate CLI with recon metrics and comparison grid"
```

---

### Task 10: 数据下载脚本 + README + 全量回归

**Files:**
- Create: `src/data/download.py`
- Create: `README.md`
- Test: `tests/test_download.py`

**Interfaces:**
- Produces:
  - `files_ready(root: str) -> bool`
  - `download_celeba(root: str) -> None` — 已存在则跳过；gdown 失败时打印手动指引并以 SystemExit(1) 退出
  - CLI `python -m src.data.download --root data/celeba`

- [ ] **Step 1: 写失败测试**

`tests/test_download.py`:

```python
from src.data.download import files_ready


def test_files_ready_false_on_empty(tmp_path):
    assert files_ready(str(tmp_path)) is False


def test_files_ready_true_when_layout_complete(tmp_path):
    (tmp_path / "img_align_celeba").mkdir()
    (tmp_path / "img_align_celeba" / "000001.jpg").write_bytes(b"x")
    (tmp_path / "list_attr_celeba.txt").write_text("x")
    (tmp_path / "list_eval_partition.txt").write_text("x")
    assert files_ready(str(tmp_path)) is True
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_download.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'src.data.download'`

- [ ] **Step 3: 实现 download.py 和 README.md**

`src/data/download.py`:

```python
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

# Google Drive 文件 ID（与 torchvision 内置源一致）
FILES = {
    "img_align_celeba.zip": "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
    "list_attr_celeba.txt": "0B7EVK8r0v71pblRyaVFSWGxPY0",
    "list_eval_partition.txt": "0B7EVK8r0v71pY0NSMzRuSzJDQkk",
}

MANUAL_HINT = """
自动下载失败（Google Drive 经常限流）。请手动下载以下文件：
  1. img_align_celeba.zip     https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
  2. list_attr_celeba.txt     https://drive.google.com/uc?id=0B7EVK8r0v71pblRyaVFSWGxPY0
  3. list_eval_partition.txt  https://drive.google.com/uc?id=0B7EVK8r0v71pY0NSMzRuSzJDQkk
将两个 txt 和 zip 放到 {root}/ 下，然后重新运行本命令（zip 会自动解压）。
""".strip()


def files_ready(root: str) -> bool:
    root = Path(root)
    img_dir = root / "img_align_celeba"
    return (
        img_dir.is_dir()
        and any(img_dir.glob("*.jpg"))
        and (root / "list_attr_celeba.txt").exists()
        and (root / "list_eval_partition.txt").exists()
    )


def _gdown(file_id: str, out_path: Path) -> bool:
    try:
        subprocess.run(
            [sys.executable, "-m", "gdown", "--fuzzy",
             f"https://drive.google.com/uc?id={file_id}", "-O", str(out_path)],
            check=True,
        )
        return out_path.exists()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_celeba(root: str) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if files_ready(str(root)):
        print(f"数据已就绪: {root}")
        return

    for name, file_id in FILES.items():
        out = root / name
        if not out.exists():
            print(f"下载 {name} ...")
            if not _gdown(file_id, out):
                raise SystemExit(MANUAL_HINT.format(root=root))

    zip_path = root / "img_align_celeba.zip"
    if zip_path.exists() and not (root / "img_align_celeba").is_dir():
        print("解压 img_align_celeba.zip（约 20 万张图，需几分钟）...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)

    if not files_ready(str(root)):
        raise SystemExit(MANUAL_HINT.format(root=root))
    print(f"数据准备完成: {root}")


def main():
    parser = argparse.ArgumentParser(description="下载并校验 CelebA 数据集")
    parser.add_argument("--root", default="data/celeba")
    args = parser.parse_args()
    download_celeba(args.root)


if __name__ == "__main__":
    main()
```

`README.md`:

````markdown
# CVAE-CelebA：属性条件人脸生成

输入人脸属性（微笑、眼镜、性别等 8 个开关），生成对应的人脸图片。
模型：卷积条件变分自编码器（CVAE），数据集：CelebA，分辨率 64×64。

## 安装

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 使用流程

```bash
# 1. 下载数据（约 1.4GB，失败时按提示手动下载）
python -m src.data.download

# 2. 训练（RTX 4060 8GB 约 1.5-2 小时；样图存 outputs/samples/）
python -m src.train --config configs/default.yaml
# 断点续训：
python -m src.train --resume checkpoints/latest.pt

# 3. 生成
python -m src.generate --attrs "Smiling=1,Eyeglasses=1" -n 8 --seed 42

# 4. 评估
python -m src.evaluate
```

可用属性（0=关 1=开）：Male, Smiling, Eyeglasses, Mustache, Blond_Hair, Black_Hair, Young, Wearing_Lipstick

## 测试

```bash
python -m pytest tests/ -v
```

## 目录结构

见 `docs/superpowers/specs/2026-07-20-cvae-celeba-design.md`。
````

- [ ] **Step 4: 运行测试 + 全量回归**

Run: `python -m pytest tests/ -v`
Expected: 全部 PASS（约 2-3 分钟，含训练冒烟）

- [ ] **Step 5: Commit**

```bash
git add src/data/download.py README.md tests/test_download.py
git commit -m "feat: CelebA download script and README"
```

---

## 完成后的人工步骤（不属于自动化任务）

1. `python -m src.data.download` 下载数据（若限流，按终端提示手动下载 3 个文件放入 `data/celeba/` 后重跑）
2. `python -m src.train` 开始训练，观察 `outputs/samples/` 中样图逐步清晰
3. 按 spec 验收标准运行 `python -m src.generate --attrs "Smiling=1,Eyeglasses=1"` 检查生成效果
