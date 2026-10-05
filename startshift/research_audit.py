"""Read-only scientific evidence audit. Does not authorize or launch training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


def group_key(record: dict) -> tuple[str, int]:
    """LIBERO-Plus task IDs are suite-local; never discard the suite."""
    return str(record["suite"]), int(record["task_id"])


def audit_metadata_contract(episodes, tasks, requested_groups):
    episodes, tasks, requested_groups = list(episodes), list(tasks), list(requested_groups)
    ids = [int(row["episode_index"]) for row in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate episode metadata IDs")
    columns = sorted(set().union(*(row.keys() for row in episodes))) if episodes else []
    requested = {group_key(row) for row in requested_groups}
    task_ids = {int(row["task_index"]) for row in tasks}
    explicit = {"robotinit_suite", "robotinit_task_id"}.issubset(columns)
    status = "UNVERIFIED_MAPPING_SOURCE" if explicit else "BLOCKED_MISSING_VARIANT_PROVENANCE"
    return {
        "status": status,
        "ready_for_matched_data_training": False,
        "verified_mapped_episodes": 0,
        "episode_count": len(ids),
        "unique_episode_count": len(set(ids)),
        "episode_columns_examined": columns,
        "dataset_task_count": len(task_ids),
        "requested_suite_qualified_groups": len(requested),
        "requested_bare_ids": len({task for _, task in requested}),
        "suite_aliases_lost_by_bare_task_id": len(requested) - len({task for _, task in requested}),
        "numeric_base_task_id_intersection": sorted(task_ids & {task for _, task in requested}),
        "episode_task_index_present": bool(episodes) and all("task_index" in r for r in episodes),
        "explicit_variant_columns_present": explicit,
        "reason": (
            "Mapping-like fields require independent source/revision verification."
            if explicit else
            "No explicit episode-to-suite/RobotInit-variant mapping in examined metadata; "
            "base-task indices and task language do not certify perturbation provenance."
        ),
    }


def audit_split_overlap(splits: dict[str, list[dict]]) -> dict:
    groups = {name: [group_key(r) for r in records] for name, records in splits.items()}
    overlaps = {}
    for left, right in combinations(groups, 2):
        common = set(groups[left]) & set(groups[right])
        if common:
            overlaps[f"{left}::{right}"] = [f"{s}:{t}" for s, t in sorted(common)]
    return {
        "sizes": {name: len(values) for name, values in groups.items()},
        "within_split_duplicate_groups": {
            name: len(values) - len(set(values)) for name, values in groups.items()
        },
        "cross_split_overlaps": overlaps,
    }


def _wilson(successes: int, n: int) -> list[float]:
    if n == 0:
        raise ValueError("Cannot summarize empty evaluation")
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def summarize_evaluations(rows, *, bootstrap_replicates=10000, bootstrap_seed=20261005):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot summarize empty evaluation")
    if bootstrap_replicates < 2:
        raise ValueError("Need at least two bootstrap replicates")
    seen = set()
    groups = defaultdict(list)
    suite_counts = defaultdict(lambda: [0, 0])
    difficulty_counts = defaultdict(lambda: [0, 0])
    for row in rows:
        if type(row["success"]) is not bool:
            raise ValueError("success must be a JSON boolean")
        key = (*group_key(row), int(row["episode"]), row.get("seed"))
        if key in seen:
            raise ValueError(f"Duplicate rollout key: {key}")
        seen.add(key)
        groups[group_key(row)].append(row["success"])
        suite_counts[str(row["suite"])][0] += int(row["success"])
        suite_counts[str(row["suite"])][1] += 1
        difficulty_counts[str(row.get("difficulty"))][0] += int(row["success"])
        difficulty_counts[str(row.get("difficulty"))][1] += 1
    rates = sorted(sum(values) / len(values) for values in groups.values())
    by_suite = defaultdict(list)
    for (suite, _), values in sorted(groups.items()):
        by_suite[suite].append((sum(values), len(values)))
    rng = random.Random(bootstrap_seed)
    bootstrap = []
    for _ in range(bootstrap_replicates):
        successes = total = 0
        for suite_groups in by_suite.values():
            for _ in suite_groups:
                s, n = rng.choice(suite_groups)
                successes += s
                total += n
        bootstrap.append(successes / total)
    successes = sum(row["success"] for row in rows)
    tail_n = max(1, math.ceil(len(rates) * 0.2))
    return {
        "n_episodes": len(rows),
        "n_pose_groups": len(groups),
        "successes": successes,
        "success_rate": successes / len(rows),
        "episode_wilson95_descriptive": _wilson(successes, len(rows)),
        "suite_stratified_pose_bootstrap95": [
            _percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975)
        ],
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_interpretation": "Exploratory, conditional on observed suite/task selection; not causal.",
        "worst_pose_success": min(rates),
        "cvar20_pose_success": sum(rates[:tail_n]) / tail_n,
        "group_trial_counts": dict(Counter(len(v) for v in groups.values())),
        "suite_results": {
            k: {"successes": s, "episodes": n, "success_rate": s / n}
            for k, (s, n) in sorted(suite_counts.items())
        },
        "difficulty_results": {
            k: {"successes": s, "episodes": n, "success_rate": s / n}
            for k, (s, n) in sorted(difficulty_counts.items())
        },
        "missing_initial_states": sum(r.get("initial_state") is None for r in rows),
        "missing_video_references": sum(not r.get("video") for r in rows),
        "unlabeled_failures": sum(
            not r["success"] and r.get("failure_type") in (None, "UNLABELED") for r in rows
        ),
        "missing_explicit_base_task": sum(
            (r.get("metadata") or {}).get("classification", {}).get("base_task") is None
            for r in rows
        ),
    }


def _read_json(path: Path):
    return json.loads(path.read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--episode-metadata", type=Path, required=True)
    parser.add_argument("--task-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing research output: {args.output}")
    import pyarrow.parquet as pq

    repo = args.repo.resolve()
    inputs = [args.episode_metadata, args.task_metadata]
    episode_file = pq.ParquetFile(args.episode_metadata)
    schema = episode_file.schema_arrow.names
    chosen = [c for c in schema if c in {
        "episode_index", "tasks", "length", "task_index", "robotinit_suite", "robotinit_task_id"
    }]
    episodes = episode_file.read(columns=chosen).to_pylist()
    tasks = pq.read_table(args.task_metadata).to_pylist()
    split_names = ["audit", "adapt_pool", "dev", "heldout_test"]
    core = {}
    for name in split_names:
        path = repo / "splits" / f"{name}.json"
        inputs.append(path)
        core[name] = _read_json(path)
    workshop = {}
    for name in ["audit_workshop_80", "adapt_pool_workshop_100", "heldout_workshop_80"]:
        path = repo / "splits" / f"{name}.json"
        inputs.append(path)
        workshop[name] = _read_json(path)
    manifest_path = repo / "splits/robotinit_manifest.json"
    inputs.append(manifest_path)
    manifest = _read_json(manifest_path)
    results = {}
    summary_matches = {}
    for name in ["id", "robotinit"]:
        path = repo / "outputs/eval/m0_workshop" / name / "eval_records.jsonl"
        summary_path = path.parent / "summary.json"
        inputs.extend([path, summary_path])
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        results[name] = summarize_evaluations(
            rows, bootstrap_replicates=args.bootstrap_replicates
        )
        old = _read_json(summary_path)
        summary_matches[name] = {
            field: math.isclose(results[name][field], old[field], abs_tol=1e-12)
            for field in ["n_episodes", "successes", "success_rate", "cvar20_pose_success"]
        }
    payload = {
        "audit_type": "READ_ONLY_REAL_METADATA_AND_HISTORICAL_RECORDS",
        "method_effect": None,
        "method_effect_status": "NOT_IDENTIFIED",
        "metadata_schema": schema,
        "episode_frame_sum": sum(row.get("length", 0) for row in episodes),
        "data_contract_all_groups": audit_metadata_contract(episodes, tasks, manifest),
        "data_contract_workshop_pool": audit_metadata_contract(
            episodes, tasks, workshop["adapt_pool_workshop_100"]
        ),
        "core_split_audit": audit_split_overlap(core),
        "workshop_split_audit": audit_split_overlap(workshop),
        "evaluation": results,
        "historical_summary_checks": summary_matches,
        "descriptive_gap_pp": 100 * (
            results["id"]["success_rate"] - results["robotinit"]["success_rate"]
        ),
        "causal_gap_identified": False,
        "causal_limit": "ID and RobotInit are not a registered paired base-task/reset comparison.",
        "inputs_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in inputs
        },
        "scientific_status": "BLOCKED_MATCHED_DATA_METHOD_TEST",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({
        "output": str(args.output), "execution": "SUCCESS",
        "scientific_status": payload["scientific_status"],
        "episodes": len(episodes), "tasks": len(tasks),
        "descriptive_gap_pp": payload["descriptive_gap_pp"],
        "method_effect": None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
