from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_pose_success(df: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    table = df.sort_values(["method", "pose_id"])
    for method, group in table.groupby("method"):
        ax.plot(range(len(group)), group["success_rate"], marker="o", label=method)
    ax.set_xlabel("Robot initial-state variant")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.set_title("Initial-state sensitivity")
    return _save(fig, path)


def plot_data_scaling(df: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for method, group in df.groupby("method"):
        group = group.sort_values("adaptation_budget")
        ax.plot(group["adaptation_budget"], group["success_rate"], marker="o", label=method)
    ax.set_xlabel("Adaptation pose/data budget")
    ax.set_ylabel("Held-out RobotInit success rate")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.set_title("Data-efficient robust adaptation")
    return _save(fig, path)


def plot_tail_reliability(df: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(df["success_rate"], df["cvar20_pose_success"])
    for _, row in df.iterrows():
        ax.annotate(str(row["method"]), (row["success_rate"], row["cvar20_pose_success"]))
    ax.set_xlabel("Mean success rate")
    ax.set_ylabel("Worst-20% pose success (CVaR)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Mean vs tail robustness")
    return _save(fig, path)


def plot_efficiency(df: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = df["trainable_parameters"].clip(lower=1)
    ax.scatter(x, df["success_rate"])
    ax.set_xscale("log")
    for _, row in df.iterrows():
        ax.annotate(str(row["method"]), (max(1, row["trainable_parameters"]), row["success_rate"]))
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Held-out RobotInit success rate")
    ax.set_ylim(0, 1.02)
    ax.set_title("Robustness vs adaptation cost")
    return _save(fig, path)
