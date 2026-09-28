#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def command(*parts: str) -> list[str]:
    return [str(x) for x in parts]


def build_plan(root: Path, budgets: list[int], seeds: list[int], config: str) -> list[list[str]]:
    plan: list[list[str]] = []
    for seed in seeds:
        for budget in budgets:
            random_split = f"splits/adapt_random_{budget}.json"
            targeted_split = f"splits/adapt_targeted_{budget}.json"
            prefix = root / f"seed_{seed}" / f"budget_{budget}"

            plan.append(command(
                "startshift", "baseline", "--config", config, "--split", random_split,
                "--method", "lora", "--output-dir", prefix / "lora_random", "--execute"
            ))
            plan.append(command(
                "startshift", "baseline", "--config", config, "--split", targeted_split,
                "--method", "lora", "--output-dir", prefix / "lora_targeted", "--execute"
            ))
            for method, cfg_name in [
                ("rise-e", "configs/rise_e.yaml"),
                ("rise-ea", "configs/rise_ea.yaml"),
                ("rise-ear", "configs/rise_ear.yaml"),
            ]:
                plan.append(command(
                    "startshift", "train", "--config", cfg_name, "--split", targeted_split,
                    "--method", method, "--output-dir", prefix / method, "--seed", str(seed)
                ))
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/matrix")
    parser.add_argument("--budgets", default="10,25,50,100")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--baseline-config", default="configs/rise_ea.yaml")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    budgets = [int(x) for x in args.budgets.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    plan = build_plan(Path(args.root), budgets, seeds, args.baseline_config)
    for idx, cmd in enumerate(plan, 1):
        print(f"[{idx:03d}/{len(plan):03d}] " + " ".join(map(str, cmd)), flush=True)
        if args.execute:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
