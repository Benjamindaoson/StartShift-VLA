from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from startshift.models.rise import RISEStateProjector, patch_smolvla_state_projector
from startshift.utils.repro import collect_environment

MANIFEST = "startshift_manifest.json"
PROJECTOR = "rise_state_projector.pt"


def save_rise_checkpoint(
    output_dir: str | Path,
    policy,
    *,
    base_policy: str,
    method: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    projector = getattr(getattr(policy, "model", None), "state_proj", None)
    if not isinstance(projector, RISEStateProjector):
        raise TypeError("save_rise_checkpoint expects a policy patched with RISE.")

    torch.save(projector.state_dict(), out / PROJECTOR)
    manifest = {
        "format_version": 1,
        "method": method,
        "base_policy": base_policy,
        "rise_projector": projector.manifest(),
        "environment": collect_environment(),
        "extra": extra or {},
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def load_rise_checkpoint(checkpoint_dir: str | Path, *, device: str | None = None):
    from lerobot.policies.smolvla import SmolVLAPolicy

    path = Path(checkpoint_dir)
    manifest = json.loads((path / MANIFEST).read_text())
    cfg = manifest["rise_projector"]
    policy = SmolVLAPolicy.from_pretrained(manifest["base_policy"])
    patch_smolvla_state_projector(policy, **cfg)
    state = torch.load(path / PROJECTOR, map_location="cpu", weights_only=True)
    policy.model.state_proj.load_state_dict(state)
    if device is not None:
        policy.to(device)
        policy.config.device = device
    policy.eval()
    return policy, manifest
