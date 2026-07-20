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
# 0. 准备数据：将 CelebA 放入 data/celeba/，目录结构应为
#    data/celeba/img_align_celeba/*.jpg
#    data/celeba/list_attr_celeba.txt
#    data/celeba/list_eval_partition.txt
#    （划分文件可按图片编号顺序生成：前 162770 张 train、接着 19867 张 val、其余 test）

# 1. 训练（RTX 4060 8GB 约 1.5-2 小时；样图存 outputs/samples/）
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
