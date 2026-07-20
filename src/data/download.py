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
            [sys.executable, "-m", "gdown",
             f"https://drive.google.com/uc?id={file_id}", "-O", str(out_path)],
            check=True,
        )
        return out_path.exists()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _generate_partition_file(root: Path) -> None:
    """官方划分按图片编号顺序：前 162770 张 train，接着 19867 张 val，其余 test。"""
    attr_path = root / "list_attr_celeba.txt"
    lines = attr_path.read_text(encoding="utf-8").splitlines()
    names = sorted(ln.split()[0] for ln in lines[2:] if ln.split())
    out = root / "list_eval_partition.txt"
    with open(out, "w", encoding="utf-8") as f:
        for i, name in enumerate(names):
            code = 0 if i < 162770 else (1 if i < 182637 else 2)
            f.write(f"{name} {code}\n")
    print(f"已从属性文件生成官方顺序划分: {out}")


def download_celeba(root: str) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if files_ready(str(root)):
        print(f"数据已就绪: {root}")
        return

    for name, file_id in FILES.items():
        out = root / name
        if out.exists():
            continue
        # 划分文件可从属性文件本地生成，不必下载
        if name == "list_eval_partition.txt" and (root / "list_attr_celeba.txt").exists():
            _generate_partition_file(root)
            continue
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
