from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from startshift.config import ExperimentConfig, load_config
from startshift.constants import OBS_STATE, STARTSHIFT_GROUP_INDEX, STARTSHIFT_INITIAL_STATE
from startshift.data.splits import load_pose_split
from startshift.models.checkpoint import save_rise_checkpoint
from startshift.models.rise import count_parameters, freeze_all_except_rise, patch_smolvla_state_projector
from startshift.training.augment import RISEBatchTransform
from startshift.training.dataset import InitialStateDataset, difficulty_group_map, load_subset_for_policy
from startshift.training.robust import GroupDRO
from startshift.utils.io import write_json
from startshift.utils.repro import collect_environment, seed_everything


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _make_policy(base_policy: str, method: str, cfg):
    from lerobot.policies.smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(base_policy)
    if method not in {"rise-e", "rise-ea", "rise-ear"}:
        raise ValueError("Custom trainer is only for rise-e, rise-ea, and rise-ear.")
    mode = "linear" if method == "rise-e" else "adapter"
    patch_smolvla_state_projector(
        policy,
        mode=mode,
        state_dim=cfg.state_dim,
        bottleneck_dim=cfg.bottleneck_dim,
        residual_scale=cfg.adapter_scale,
        freeze_base=cfg.freeze_base_state_proj,
    )
    trainable = freeze_all_except_rise(policy)
    if not trainable:
        raise RuntimeError("RISE patch produced no trainable parameters.")
    return policy


def train_rise(
    config: str | Path | ExperimentConfig,
    *,
    split: str | Path,
    method: str | None = None,
    limit_per_task: int | None = None,
    output_dir: str | Path | None = None,
    seed: int | None = None,
) -> Path:
    cfg = load_config(config) if not isinstance(config, ExperimentConfig) else config
    method = method or cfg.policy.method
    if seed is not None:
        cfg.train.seed = seed
    if output_dir is not None:
        cfg.train.output_dir = str(output_dir)
    seed_everything(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    policy = _make_policy(cfg.policy.base_policy, method, cfg.policy)
    policy.to(device)
    policy.config.device = str(device)

    records = load_pose_split(split)
    task_ids = [record.task_id for record in records]
    dataset, episodes = load_subset_for_policy(
        repo_id=cfg.data.dataset_repo,
        policy_config=policy.config,
        task_ids=task_ids,
        limit_per_task=limit_per_task,
    )
    dataset = InitialStateDataset(dataset, difficulty_group_map(records))

    from lerobot.policies import make_pre_post_processors
    from lerobot.utils.collate import lerobot_collate_fn

    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=cfg.policy.base_policy,
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    rise_transform = RISEBatchTransform(dataset.meta.stats[OBS_STATE])
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=lerobot_collate_fn if dataset.meta.has_language_columns else None,
        persistent_workers=cfg.train.num_workers > 0,
    )
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    dro = GroupDRO(num_groups=5, device=device) if method == "rise-ear" else None

    out = Path(cfg.train.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(collect_environment(), out / "environment.json")
    write_json(
        {
            "method": method,
            "split": str(split),
            "task_ids": task_ids,
            "episodes": episodes,
            "parameters": count_parameters(policy),
        },
        out / "run_manifest.json",
    )
    log_path = out / "train.jsonl"
    iterator = iter(dataloader)
    started = time.perf_counter()
    policy.train()

    for step in trange(1, cfg.train.steps + 1, desc=f"training {method}"):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            raw = next(iterator)
        initial_raw = raw.pop(STARTSHIFT_INITIAL_STATE)
        groups = raw.pop(STARTSHIFT_GROUP_INDEX).to(device)
        batch = preprocessor(raw)
        batch = rise_transform(batch, initial_raw)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, cfg.train.bf16):
            if dro is None:
                loss, details = policy(batch)
                robust_stats = None
            else:
                per_sample, details = policy(batch, reduction="none")
                loss, robust_stats = dro(per_sample, groups)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip_norm)
        optimizer.step()

        if step % cfg.train.log_every == 0 or step == 1:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": float(grad_norm.detach().cpu()),
                "elapsed_s": time.perf_counter() - started,
                "details": details,
                "robust": robust_stats,
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        if step % cfg.train.save_every == 0 or step == cfg.train.steps:
            save_rise_checkpoint(
                out / f"checkpoint-{step:08d}",
                policy,
                base_policy=cfg.policy.base_policy,
                method=method,
                extra={"step": step, "split": str(split)},
            )

    save_rise_checkpoint(
        out / "last",
        policy,
        base_policy=cfg.policy.base_policy,
        method=method,
        extra={"step": cfg.train.steps, "split": str(split), "final": True},
    )
    write_json(
        {"elapsed_s": time.perf_counter() - started, "final_step": cfg.train.steps, "method": method},
        out / "done.json",
    )
    return out
