"""GPU detection so XGBoost/TensorFlow use CUDA when available (e.g. Colab T4)."""
import shutil
import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def has_cuda():
    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def xgb_device():
    return "cuda" if has_cuda() else "cpu"
