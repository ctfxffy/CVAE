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
