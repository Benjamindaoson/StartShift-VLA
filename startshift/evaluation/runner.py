from __future__ import annotations

from pathlib import Path

import torch

from startshift.data.splits import load_pose_split
from startshift.evaluation.metrics import summarize_records
from startshift.models.checkpoint import MANIFEST, load_rise_checkpoint
from startshift.training.augment import StartShiftInferencePreprocessor
from startshift.types import EvalRecord, PoseRecord
from startshift.utils.io import write_json, write_jsonl
from startshift.utils.repro import collect_environment, seed_everything


def _load_policy(policy_path: str, *, device: str):
    checkpoint = Path(policy_path)
    if checkpoint.is_dir() and (checkpoint / MANIFEST).exists():
        policy, manifest = load_rise_checkpoint(checkpoint, device=device)
        processor_source = manifest["base_policy"]
        is_rise = True
        method = manifest["method"]
    else:
        from lerobot.policies.smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(policy_path)
        policy.to(device)
        policy.config.device = device
        processor_source = policy_path
        is_rise = False
        method = "base"
    policy.eval()
    return policy, processor_source, is_rise, method


def _episode_steps(done: torch.Tensor) -> list[int]:
    values: list[int] = []
    for row in done.bool():
        indices = torch.nonzero(row, as_tuple=False)
        values.append(int(indices[0].item() + 1) if len(indices) else int(row.numel()))
    return values


def evaluate_records(
    *,
    policy_path: str,
    records: list[PoseRecord],
    output_dir: str | Path,
    episodes_per_task: int = 10,
    seed: int = 42,
    device: str = "cuda",
    method_name: str | None = None,
    hard_reset: bool = True,
    init_states: bool = True,
    env_type: str = "libero_plus",
    metadata: dict | None = None,
    record_trajectories: bool = False,
) -> Path:
    """Evaluate records while preserving the perturbation-task as the pose unit.

    LIBERO-Plus task IDs encode perturbation variants. Multiple rollout episodes
    for a task are repeated trials of the same perturbation, not independent
    poses. Keeping one pose_id per task is required for meaningful worst-pose
    and CVaR metrics.
    """
    from lerobot.envs.configs import LiberoEnv, LiberoPlusEnv
    from lerobot.policies import make_pre_post_processors
    from lerobot.scripts.lerobot_eval import rollout

    if env_type == "libero":
        from startshift.evaluation.libero_id import install_vanilla_libero_benchmarks

        install_vanilla_libero_benchmarks()

    if not torch.cuda.is_available() and device.startswith("cuda"):
        device = "cpu"
    seed_everything(seed)
    policy, processor_source, is_rise, inferred_method = _load_policy(policy_path, device=device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=processor_source,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    if is_rise:
        preprocessor = StartShiftInferencePreprocessor(preprocessor)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[EvalRecord] = []
    env_cls = LiberoPlusEnv if env_type == "libero_plus" else LiberoEnv

    for record_index, record in enumerate(records):
        env_cfg = env_cls(
            task=record.suite,
            task_ids=[record.task_id],
            hard_reset=hard_reset,
            init_states=init_states,
            max_parallel_tasks=1,
        )
        env_preprocessor, env_postprocessor = env_cfg.get_env_processors()
        env_map = env_cfg.create_envs(n_envs=episodes_per_task, use_async_envs=False)
        env = env_map[record.suite][record.task_id]
        recording_dir = None
        if record_trajectories:
            recording_dir = output / "recordings" / record.suite / f"task_{record.task_id:05d}"
        try:
            rollout_data = rollout(
                env,
                policy,
                env_preprocessor,
                env_postprocessor,
                preprocessor,
                postprocessor,
                seeds=[seed + record_index * 10_000 + i for i in range(episodes_per_task)],
                return_observations=False,
                recording_dir=recording_dir,
                env_features=env_cfg.features if recording_dir is not None else None,
            )
        finally:
            env.close()

        successes = rollout_data["success"].bool().any(dim=1).cpu().tolist()
        steps = _episode_steps(rollout_data["done"].cpu())
        rewards = rollout_data["reward"].sum(dim=1).cpu().tolist()
        for episode, success in enumerate(successes):
            all_rows.append(
                EvalRecord(
                    method=method_name or inferred_method,
                    suite=record.suite,
                    task_id=record.task_id,
                    episode=episode,
                    success=bool(success),
                    pose_id=record.group_id,
                    difficulty=record.difficulty,
                    steps=steps[episode],
                    reward=float(rewards[episode]),
                    failure_type=None if success else "UNLABELED",
                    seed=seed + record_index * 10_000 + episode,
                    metadata={
                        "classification": record.to_dict(),
                        "env_type": env_type,
                        "init_states": init_states,
                        "hard_reset": hard_reset,
                        "recording_dir": str(recording_dir) if recording_dir else None,
                    },
                )
            )

    records_path = output / "eval_records.jsonl"
    write_jsonl([row.to_dict() for row in all_rows], records_path)
    summary = summarize_records(all_rows)
    summary["method"] = method_name or inferred_method
    summary["policy_path"] = policy_path
    summary["env_type"] = env_type
    summary["init_states"] = init_states
    summary["hard_reset"] = hard_reset
    summary["metadata"] = metadata or {}
    summary["environment"] = collect_environment()
    write_json(summary, output / "summary.json")
    return records_path


def evaluate_split(
    *,
    policy_path: str,
    split_path: str | Path,
    output_dir: str | Path,
    episodes_per_task: int = 10,
    seed: int = 42,
    device: str = "cuda",
    method_name: str | None = None,
    hard_reset: bool = True,
    init_states: bool = True,
    metadata: dict | None = None,
    record_trajectories: bool = False,
) -> Path:
    return evaluate_records(
        policy_path=policy_path,
        records=load_pose_split(split_path),
        output_dir=output_dir,
        episodes_per_task=episodes_per_task,
        seed=seed,
        device=device,
        method_name=method_name,
        hard_reset=hard_reset,
        init_states=init_states,
        env_type="libero_plus",
        metadata=metadata,
        record_trajectories=record_trajectories,
    )


def evaluate_id_suites(
    *,
    policy_path: str,
    suites: list[str],
    output_dir: str | Path,
    tasks_per_suite: int = 10,
    episodes_per_task: int = 10,
    seed: int = 42,
    device: str = "cuda",
    method_name: str | None = None,
) -> Path:
    records: list[PoseRecord] = []
    for suite in suites:
        for task_id in range(tasks_per_suite):
            records.append(
                PoseRecord(
                    suite=suite,
                    task_id=task_id,
                    category="id",
                    pose_id=f"{suite}:{task_id}",
                )
            )
    return evaluate_records(
        policy_path=policy_path,
        records=records,
        output_dir=output_dir,
        episodes_per_task=episodes_per_task,
        seed=seed,
        device=device,
        method_name=method_name,
        env_type="libero",
        metadata={"split": "id"},
    )
