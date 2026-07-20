# CVAE-CelebA：属性条件人脸生成

输入人脸属性（性别、微笑、眼镜等 8 个 0/1 开关），生成对应的人脸图片。

- **模型**: 卷积条件变分自编码器（CVAE），latent_dim=256，参数量约 11.9M
- **预训练权重**: 从 [Hugging Face](https://huggingface.co/GXY12345/cvae_celebA) 下载 `best.pt`，放到 `checkpoints/` 目录
- **数据集**: CelebA，img_align_celeba（202,599 张人脸，分辨率 64×64）


---


## 数据准备

将 CelebA 数据集放入 `data/celeba/`，目录结构如下：

```
data/celeba/
├── img_align_celeba/          # 202,599 张 .jpg 图片（178×218）
├── list_attr_celeba.txt       # 40 个属性的标注文件
└── list_eval_partition.txt    # train/val/test 划分文件
```

> 数据集未下载可参考 [CelebA 官网](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 或 Hugging Face 镜像。

---

## 使用流程

### 1. 训练

```bash
python -m src.train --config configs/default.yaml
```

断点续训：

```bash
python -m src.train --resume checkpoints/latest.pt
```

训练过程中：
- 终端显示 tqdm 进度条（实时 loss + 全局步数）
- 每 500 步在 `outputs/samples/step_XXXXXX.png` 存一张**样本网格图**（见下方"如何观察训练效果"）
- 每个 epoch 保存一次 checkpoint（`checkpoints/latest.pt`，val loss 最优时额外存 `best.pt`）

### 2. 生成

```bash
python -m src.generate --attrs "Smiling=1,Eyeglasses=1" -n 8 --seed 42
```

选项：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--attrs` | 全 0 | 属性组合，如 `"Male=1,Smiling=0"`，逗号分隔 |
| `-n` | 8 | 生成张数 |
| `--seed` | 42 | 随机种子 |
| `--checkpoint` | `checkpoints/best.pt` | 模型权重路径 |
| `--config` | `configs/default.yaml` | 配置文件路径 |
| `--out` | 自动命名 | 指定输出路径，默认 `outputs/generated/<attrs>_n<n>_seed<seed>.png` |

### 3. 评估

```bash
python -m src.evaluate
```

输出：

- **终端**: val 集前 20 个 batch 的平均 MSE 与 KL 散度
- **图片**: `outputs/eval/recon_grid.png`（上排原始图片、下排对应重建，对比评估重建质量）

---

## 可用属性

共 8 个（按 `configs/default.yaml` 中 `attr_names` 顺序），值取 0（关）或 1（开）：

| 属性 | 中文含义 | 属性 | 中文含义 |
|---|---|---|---|
| Male | 男性 | Black_Hair | 黑发 |
| Smiling | 微笑 | Young | 年轻 |
| Eyeglasses | 眼镜 | Wearing_Lipstick | 涂口红 |
| Mustache | 胡子 | Blond_Hair | 金发 |

---



## 配置说明

`configs/default.yaml` 中可调的参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `seed` | 42 | 全局随机种子 |
| `data_root` | `data/celeba` | 数据集目录 |
| `image_size` | 64 | 输入分辨率 |
| `attr_names` | 8 个属性 | 可在列表中增减属性名（须与 CelebA 官方一致） |
| `latent_dim` | 256 | 隐变量 z 维度 |
| `beta` | 1.0 | KL 散度权重（>1 属性控制更强但图更模糊，<1 反之） |
| `batch_size` | 128 | 显存不足时减半 |
| `num_workers` | 4 | DataLoader 线程数 |
| `lr` | 0.001 | Adam 学习率 |
| `epochs` | 25 | 训练轮数 |
| `sample_every` | 500 | 样本图保存间隔（步数） |
| `checkpoint_dir` | `checkpoints` | 模型保存目录 |
| `output_dir` | `outputs` | 输出目录 |

---

## 项目结构

```
E:\CVAE\
├── configs/default.yaml       # 所有超参数
├── data/celeba/               # 数据集（不入库）
├── src/
│   ├── data/dataset.py        # CelebA 数据集封装（CenterCrop→Resize→归一化，属性 0/1）
│   ├── models/
│   │   ├── encoder.py         # 卷积编码器：image + attr → μ, logσ²
│   │   ├── decoder.py         # 转置卷积解码器：z + attr → image
│   │   └── cvae.py            # CVAE 组合：重参数化、前向、损失、采样
│   ├── training/
│   │   ├── trainer.py         # 训练循环：AMP、tqdm、定期采样、checkpoint
│   │   └── utils.py           # 工具：种子、checkpoint IO、图像保存
│   ├── train.py               # 训练入口
│   ├── generate.py            # 生成入口
│   └── evaluate.py            # 评估入口
├── checkpoints/               # 模型权重（不入库）
├── outputs/                   # 生成结果（不入库）
└── requirements.txt
```

---

## 典型 workflow

```bash
# 0. 确认 GPU 可用
python -c "import torch; assert torch.cuda.is_available()"

# 1. 训练
python -m src.train
# 期间观察 outputs/samples/ 中样图逐步清晰

# 2. 用最优 checkpoint 生成
python -m src.generate --attrs "Smiling=1,Eyeglasses=1,Blond_Hair=1" -n 8

# 3. 尝试不同属性组合验证效果
python -m src.generate --attrs "Male=1,Mustache=1" -n 8
python -m src.generate --attrs "Young=1,Black_Hair=1" -n 8

# 4. 查看重建质量
python -m src.evaluate
```
