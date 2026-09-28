#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def _cmd(parts):
    return [str(x) for x in parts]


def build_plan(config_path: str | Path) -> list[list[str]]:
    cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    heldout_split = cfg.get("heldout_split", "splits/heldout_test.json")
    episodes = int(cfg.get("episodes", 10))
    device = str(cfg.get("device", "cuda"))
    root = Path(cfg.get("output_root", "outputs/eval/matrix"))

    plan: list[list[str]] = []
    for run in cfg.get("runs", []):
        required = {"name", "policy", "seed", "adaptation_budget"}
        missing = required - set(run)
        if missing:
            raise ValueError(f"Evaluation run missing fields: {sorted(missing)}")

        name = str(run["name"])
        policy = str(run["policy"])
        seed = int(run["seed"])
        budget = int(run["adaptation_budget"])
        trainable = int(run.get("trainable_parameters", 0))
        gpu_hours = run.get("gpu_hours")
        target = root / f"seed_{seed}" / f"budget_{budget}" / name

        heldout_cmd = [
            "startshift",
            "eval",
            "--policy",
            policy,
            "--split",
            heldout_split,
            "--output-dir",
            target / "heldout",
            "--episodes",
            str(episodes),
            "--seed",
            str(seed),
            "--device",
            device,
            "--method-name",
            name,
            "--adaptation-budget",
            str(budget),
            "--trainable-parameters",
            str(trainable),
        ]
        if gpu_hours is not None:
            heldout_cmd += ["--gpu-hours", str(gpu_hours)]
        if bool(run.get("record_trajectories", False)):
            heldout_cmd += ["--record-trajectories"]
        plan.append(_cmd(heldout_cmd))

        plan.append(
            _cmd(
                [
                    "startshift",
                    "eval-id",
                    "--policy",
                    policy,
                    "--output-dir",
                    target / "id",
                    "--episodes",
                    str(episodes),
                    "--seed",
                    str(seed),
                    "--device",
                    device,
                    "--method-name",
                    name,
                ]
            )
        )
    if not plan:
        raise ValueError("Evaluation matrix contains no runs")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute explicit StartShift policy paths on held-out RobotInit and ID suites"
    )
    parser.add_argument("--config", default="configs/eval_matrix.example.yaml")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    plan = build_plan(args.config)
    for idx, cmd in enumerate(plan, 1):
        print(f"[{idx:03d}/{len(plan):03d}] " + " ".join(cmd), flush=True)
        if args.execute:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
