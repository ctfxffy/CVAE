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
