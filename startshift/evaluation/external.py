from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from startshift.data.splits import load_pose_split


def official_eval_args(
    *,
    policy_path: str,
    split_path: str | Path,
    output_dir: str | Path,
    episodes: int = 10,
    batch_size: int = 1,
) -> list[list[str]]:
    """Build official lerobot-eval commands for arbitrary LeRobot policies.

    This path is deliberately model-agnostic and is used to verify that the
    StartShift phenomenon is not specific to SmolVLA.
    """
    records = load_pose_split(split_path)
    by_suite: dict[str, list[int]] = {}
    for record in records:
        by_suite.setdefault(record.suite, []).append(record.task_id)

    commands: list[list[str]] = []
    root = Path(output_dir)
    for suite, ids in sorted(by_suite.items()):
        target = root / suite
        commands.append(
            [
                "lerobot-eval",
                f"--policy.path={policy_path}",
                "--env.type=libero_plus",
                f"--env.task={suite}",
                f"--env.task_ids={json.dumps(sorted(set(ids)))}",
                f"--eval.n_episodes={episodes}",
                f"--eval.batch_size={batch_size}",
                f"--output_dir={target}",
            ]
        )
    return commands


def run_official_eval(commands: list[list[str]], *, execute: bool = True) -> list[str]:
    shell = [" ".join(shlex.quote(part) for part in command) for command in commands]
    if execute:
        for command in commands:
            subprocess.run(command, check=True)
    return shell
