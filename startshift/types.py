from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PoseRecord:
    """One LIBERO-Plus perturbation task / pose variant.

    The benchmark's exact JSON schema has changed across releases.  StartShift
    therefore normalizes upstream metadata into this stable record.
    """

    suite: str
    task_id: int
    category: str
    difficulty: str | int | None = None
    base_task: str | int | None = None
    pose_id: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def group_id(self) -> str:
        return self.pose_id or f"{self.suite}:{self.task_id}"

    @property
    def stratum(self) -> str:
        return f"{self.suite}|{self.base_task}|{self.difficulty}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PoseRecord:
        known = {k: value.get(k) for k in ("suite", "task_id", "category", "difficulty", "base_task", "pose_id")}
        known["task_id"] = int(known["task_id"])
        known["metadata"] = value.get("metadata")
        return cls(**known)


@dataclass(slots=True)
class EvalRecord:
    """Normalized per-rollout evaluation record."""

    method: str
    suite: str
    task_id: int
    episode: int
    success: bool
    pose_id: str | None = None
    difficulty: str | int | None = None
    initial_state: list[float] | None = None
    steps: int | None = None
    reward: float | None = None
    failure_type: str | None = None
    video: str | None = None
    seed: int | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvalRecord:
        return cls(**value)
