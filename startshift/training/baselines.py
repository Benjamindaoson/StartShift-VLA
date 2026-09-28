from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from startshift.data.splits import load_pose_split
from startshift.training.dataset import episodes_for_task_ids


def resolve_episode_subset(
    *, dataset_repo: str, split_path: str | Path, limit_per_task: int | None = None
) -> list[int]:
    from lerobot.datasets import LeRobotDatasetMetadata

    records = load_pose_split(split_path)
    meta = LeRobotDatasetMetadata(dataset_repo)
    return episodes_for_task_ids(meta, [r.task_id for r in records], limit_per_task=limit_per_task)


def lerobot_train_args(
    *,
    method: str,
    base_policy: str,
    dataset_repo: str,
    episodes: list[int],
    output_dir: str,
    steps: int,
    batch_size: int,
    lr: float,
) -> list[str]:
    args = [
        "lerobot-train",
        f"--policy.path={base_policy}",
        f"--dataset.repo_id={dataset_repo}",
        f"--dataset.episodes={json.dumps(episodes)}",
        f"--output_dir={output_dir}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--policy.optimizer_lr={lr}",
        "--policy.output_features=null",
        "--policy.input_features=null",
    ]
    if method == "lora":
        args += ["--peft.method_type=LORA", "--peft.r=64", "--peft.lora_alpha=64"]
    elif method in {"standard-ft", "expert-ft"}:
        pass
    else:
        raise ValueError(f"Unsupported baseline method: {method}")
    return args


def run_baseline(
    *,
    method: str,
    base_policy: str,
    dataset_repo: str,
    split_path: str | Path,
    output_dir: str,
    steps: int,
    batch_size: int,
    lr: float,
    limit_per_task: int | None = None,
    execute: bool = True,
) -> list[str]:
    episodes = resolve_episode_subset(
        dataset_repo=dataset_repo, split_path=split_path, limit_per_task=limit_per_task
    )
    args = lerobot_train_args(
        method=method,
        base_policy=base_policy,
        dataset_repo=dataset_repo,
        episodes=episodes,
        output_dir=output_dir,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
    )
    if execute:
        subprocess.run(args, check=True)
    return args


def shell_command(args: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in args)
