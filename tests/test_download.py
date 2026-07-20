from src.data.download import files_ready


def test_files_ready_false_on_empty(tmp_path):
    assert files_ready(str(tmp_path)) is False


def test_files_ready_true_when_layout_complete(tmp_path):
    (tmp_path / "img_align_celeba").mkdir()
    (tmp_path / "img_align_celeba" / "000001.jpg").write_bytes(b"x")
    (tmp_path / "list_attr_celeba.txt").write_text("x")
    (tmp_path / "list_eval_partition.txt").write_text("x")
    assert files_ready(str(tmp_path)) is True
