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
