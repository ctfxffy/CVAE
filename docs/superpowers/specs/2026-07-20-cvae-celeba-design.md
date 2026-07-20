# CVAE-CelebA 属性条件人脸生成 — 设计文档

日期：2026-07-20
状态：已确认

## 1. 目标与验收标准

搭建完整神经网络项目，用 PyTorch 在 CelebA 数据集上训练条件变分自编码器（CVAE），实现"输入常见人脸属性 → 生成对应人脸图片"。

**验收标准**：
- `python -m src.generate --attrs "Smiling=1,Eyeglasses=1" -n 8` 生成 8 张人脸，肉眼可辨微笑且戴眼镜
- 更换属性组合，生成结果有对应可见变化
- 验证集重建 loss 收敛平稳

**环境约束**：Windows 11，Python 3.10.2，RTX 4060 Laptop 8GB 显存。

**范围（YAGNI）**：命令行交付（训练/生成/评估三个入口），不做 Web UI，不做 GAN/diffusion 等更高画质方案。

## 2. 技术选型

- 框架：PyTorch + torchvision
- 数据集：CelebA（img_align_celeba + list_attr_celeba.txt），脚本自动下载
- 条件属性（8 个，可在 yaml 修改）：`Male, Smiling, Eyeglasses, Mustache, Blond_Hair, Black_Hair, Young, Wearing_Lipstick`
- 分辨率：64×64
- 架构：卷积 CVAE（方案 A），latent_dim=256

## 3. 项目结构

```
E:\CVAE\
├── configs/
│   └── default.yaml          # 所有超参数集中处
├── data/celeba/              # 数据集（gitignore）
├── src/
│   ├── data/
│   │   ├── download.py       # 下载 + 校验 CelebA（含镜像回退）
│   │   └── dataset.py        # 数据集封装：输出 (image 3×64×64, attr 8维0/1)
│   ├── models/
│   │   ├── encoder.py        # 卷积编码器：image + attr → μ, logσ²
│   │   ├── decoder.py        # 转置卷积解码器：z + attr → image
│   │   └── cvae.py           # 组合 + reparameterize + loss + sample
│   ├── training/
│   │   ├── trainer.py        # 训练循环：AMP、checkpoint、定期采样
│   │   └── utils.py          # 日志、种子、checkpoint IO、网格图保存
│   ├── train.py              # 入口：python -m src.train --config configs/default.yaml
│   ├── generate.py           # 入口：python -m src.generate --attrs "Male=1,Smiling=1" -n 8
│   └── evaluate.py           # 入口：重建质量 + 属性条件一致性抽样
├── tests/                    # pytest
├── checkpoints/              # 权重（gitignore）
├── outputs/                  # 生成样图（gitignore）
└── requirements.txt
```

**模块职责**：
- `dataset.py`：唯一的数据来源入口，输出 `(3×64×64 float 图像归一化到 [-1,1], 8 维 0/1 属性向量)`
- `encoder.py` / `decoder.py`：纯网络结构；属性注入方式只在这两处定义
- `cvae.py`：组合编解码器并定义 loss，对外暴露 `forward(x, attr) → (x̂, μ, logσ²)` 与 `sample(attr, n)`
- `trainer.py`：唯一训练循环，依赖注入 model/dataloader/config
- 三个入口脚本为薄层，只做参数解析与调用

## 4. 模型架构

### 编码器（image + attr → μ, logσ²）

```
输入图像 3×64×64
attr(8) → FC → 64×64 → reshape 1×64×64，与图像通道拼接 → 4×64×64
Conv 4→64,   k4 s2 p1 → 64×32×32    (BN + LeakyReLU)
Conv 64→128, k4 s2 p1 → 128×16×16   (BN + LeakyReLU)
Conv 128→256,k4 s2 p1 → 256×8×8     (BN + LeakyReLU)
Conv 256→512,k4 s2 p1 → 512×4×4     (BN + LeakyReLU)
展平 8192 → FC → μ(256), logσ²(256)
```

### 重参数化

`z = μ + σ·ε, ε~N(0,1)`；训练可导，推理时直接 z~N(0,1)。

### 解码器（z + attr → image）

```
z(256) 拼接 attr(8) → 264 → FC → 8192 → reshape 512×4×4
ConvT 512→256, k4 s2 p1 → 256×8×8    (BN + ReLU)
ConvT 256→128, k4 s2 p1 → 128×16×16  (BN + ReLU)
ConvT 128→64,  k4 s2 p1 → 64×32×32   (BN + ReLU)
ConvT 64→3,    k4 s2 p1 → 3×64×64, Tanh → [-1,1]
```

### 损失函数

`L = MSE(x, x̂) + β·KL(q(z|x,c) ‖ N(0,1))`，β 默认 1.0，yaml 可调。

参数量约 2000 万。

## 5. 数据流

1. `download.py` 下载 img_align_celeba.zip 与 list_attr_celeba.txt，校验文件完整性；torchvision 的 Google Drive 源限流时回退镜像 URL，再失败则打印手动放置指引
2. `dataset.py`：读图（img_align_celeba 原始尺寸 178×218）→ CenterCrop 到 148×148（去掉上下背景冗余，避免直接缩放导致人脸纵向压扁）→ Resize 64×64 → ToTensor → 归一化 [-1,1]；属性从 txt 读 8 列，{-1,1} 映射 {0,1}
3. DataLoader：batch=128，官方划分 train 162770 / val 19867（test 划分不用）
4. 训练中每 500 步用固定 z + 8 组属性组合采样存图到 `outputs/samples/`
5. 生成：从 N(0,1) 采 n 个 z，拼接用户指定属性向量，解码，保存网格图

## 6. 训练配置（configs/default.yaml）

| 项目 | 值 |
|---|---|
| image_size | 64 |
| latent_dim | 256 |
| batch_size | 128 |
| optimizer | Adam, lr=1e-3 |
| β | 1.0 |
| epochs | 30（约 3.8 万步，预计 1.5–2 小时） |
| AMP 混合精度 | 开启 |
| 采样频率 | 每 500 步 |
| checkpoint | 每 epoch 存 latest + 保留最优 val loss；支持 `--resume` 续训 |

## 7. 错误处理

1. **下载失败**：torchvision → 镜像 URL → 手动指引（目录、所需文件清单）三级策略；已有文件则校验后跳过
2. **坏图**：dataset 捕获加载异常，记录并跳过该样本
3. **CUDA OOM**：捕获后提示将 batch_size 减半重试
4. **checkpoint 不匹配**：加载时校验 key 与 shape，失败给出明确错误信息
5. **生成参数错误**：`generate.py` 校验属性名与取值（0/1），出错时列出合法属性清单

## 8. 测试（pytest）

- `test_models.py`：encoder/decoder/cvae 的 shape 断言（3×64×64 → μ,logσ² 各 256 维；z+attr → 3×64×64）；reparameterize 梯度可通
- `test_loss.py`：随机数据上 loss 为正且有限；单 batch 过拟合数步 loss 下降（训练链路冒烟）
- `test_dataset.py`：用临时假图片目录验证输出 shape、归一化范围 [-1,1]、属性 0/1
- 不测：真实数据下载、完整训练收敛（由 val loss 与样图人工判断）
