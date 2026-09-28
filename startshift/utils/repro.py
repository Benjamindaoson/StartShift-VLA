from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from typing import Any

import numpy as np


def _git_sha(cwd: str | None = None) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "project_git_sha": _git_sha(),
    }
    try:
        import torch

        result.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                "gpu_count": torch.cuda.device_count(),
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            }
        )
    except ImportError:
        result["torch"] = None

    for name, path in {
        "lerobot_git_sha": "third_party/lerobot",
        "libero_plus_git_sha": "third_party/LIBERO-plus",
    }.items():
        result[name] = _git_sha(path)
    return result


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
