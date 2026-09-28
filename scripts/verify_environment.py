#!/usr/bin/env python
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _read_lock() -> dict[str, str]:
    import yaml

    data = yaml.safe_load((ROOT / "third_party.lock").read_text())
    return {
        "lerobot": str(data["lerobot"]["commit"]),
        "libero_plus": str(data["libero_plus"]["commit"]),
    }


def _require_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise RuntimeError(f"Failed to import {name}: {exc}") from exc


def main() -> int:
    if sys.version_info < (3, 12):
        raise RuntimeError(
            f"StartShift robotics runtime requires Python >=3.12; got {sys.version.split()[0]}. "
            "Run bash scripts/bootstrap.sh, then source scripts/env.sh."
        )

    lock = _read_lock()
    checks: dict[str, object] = {
        "python": sys.version,
        "pinned": lock,
        "third_party": {},
        "imports": {},
    }

    for key, folder in [
        ("lerobot", ROOT / "third_party" / "lerobot"),
        ("libero_plus", ROOT / "third_party" / "LIBERO-plus"),
    ]:
        head = _git_head(folder)
        checks["third_party"][key] = {"path": str(folder), "head": head, "expected": lock[key]}
        if head is None:
            raise RuntimeError(f"{folder} is not cloned. Run scripts/bootstrap.sh")
        if head != lock[key]:
            raise RuntimeError(f"{key} revision mismatch: {head} != {lock[key]}")

    lerobot = _require_import("lerobot")
    libero = _require_import("libero")
    torch = _require_import("torch")
    checks["imports"] = {
        "lerobot": getattr(lerobot, "__file__", None),
        "libero": getattr(libero, "__file__", None),
        "torch": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }

    from lerobot.envs.configs import LiberoPlusEnv
    from lerobot.policies.smolvla import SmolVLAPolicy

    checks["api"] = {
        "LiberoPlusEnv": f"{LiberoPlusEnv.__module__}.{LiberoPlusEnv.__name__}",
        "SmolVLAPolicy": f"{SmolVLAPolicy.__module__}.{SmolVLAPolicy.__name__}",
    }

    print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
