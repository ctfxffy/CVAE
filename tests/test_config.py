from src.training.utils import load_config


def test_load_config_returns_expected_keys():
    cfg = load_config("configs/default.yaml")
    assert cfg["latent_dim"] == 256
    assert cfg["image_size"] == 64
    assert cfg["batch_size"] == 128
    assert len(cfg["attr_names"]) == 8
    assert cfg["attr_names"][0] == "Male"
