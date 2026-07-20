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
