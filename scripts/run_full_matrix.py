#![LOCAL_PATH] python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def command(*parts: str | Path) -> list[str]:
    return [str(x) for x in parts]


def build_plan(
    root: Path,
    budgets: list[int],
    seeds: list[int],
    config: str,
    *,
    include_standard_ft: bool = False,
    include_expert_ft: bool = True,
    include_tail: bool = False,
) -> list[list[str]]:
    """Build the matched-data training matrix.

    The same random/targeted split is reused across competing methods for a
    given budget. Formal evaluation is intentionally a separate stage because
    LeRobot and RISE checkpoint layouts differ; this script never guesses a
    checkpoint path.
    """
    plan: list[list[str]] = []
    for seed in seeds:
        for budget in budgets:
            random_split = f"splits/adapt_random_{budget}.json"
            targeted_split = f"splits/adapt_targeted_{budget}.json"
            prefix = root / f"seed_{seed}" / f"budget_{budget}"

            baseline_methods = ["lora"]
            if include_expert_ft:
                baseline_methods.append("expert-ft")
            if include_standard_ft:
                baseline_methods.append("standard-ft")

            for baseline in baseline_methods:
                for split_name, split_path in [
                    ("random", random_split),
                    ("targeted", targeted_split),
                ]:
                    plan.append(
                        command(
                            "startshift",
                            "baseline",
                            "--config",
                            config,
                            "--split",
                            split_path,
                            "--method",
                            baseline,
                            "--output-dir",
                            prefix / f"{baseline}_{split_name}",
                            "--seed",
                            str(seed),
                            "--execute",
                        )
                    )

            rise_methods = [
                ("rise-e", "configs/rise_e.yaml"),
                ("rise-ea", "configs/rise_ea.yaml"),
            ]
            if include_tail:
                rise_methods.append(("rise-ear", "configs/rise_ear.yaml"))

            for method, cfg_name in rise_methods:
                plan.append(
                    command(
                        "startshift",
                        "train",
                        "--config",
                        cfg_name,
                        "--split",
                        targeted_split,
                        "--method",
                        method,
                        "--output-dir",
                        prefix / method,
                        "--seed",
                        str(seed),
                    )
                )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate/execute the matched-data StartShift training matrix")
    parser.add_argument("--root", default="outputs/matrix")
    parser.add_argument("--budgets", default="10,25,50,100")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--baseline-config", default="configs/rise_ea.yaml")
    parser.add_argument("--include-standard-ft", action="store_true")
    parser.add_argument("--skip-expert-ft", action="store_true")
    parser.add_argument("--include-tail", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    budgets = [int(x) for x in args.budgets.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    plan = build_plan(
        Path(args.root),
        budgets,
        seeds,
        args.baseline_config,
        include_standard_ft=args.include_standard_ft,
        include_expert_ft=not args.skip_expert_ft,
        include_tail=args.include_tail,
    )
    for idx, cmd in enumerate(plan, 1):
        print(f"[{idx:03d}/{len(plan):03d}] " + " ".join(map(str, cmd)), flush=True)
        if args.execute:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
