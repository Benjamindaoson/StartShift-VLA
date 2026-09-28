#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess

from startshift.evaluation.matrix import build_plan


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
