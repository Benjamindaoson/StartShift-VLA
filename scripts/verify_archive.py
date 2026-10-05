"""Verify historical evidence offline; never import a model or launch an experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def summarize(rows):
    if not rows:
        raise ValueError("Empty evaluation")
    seen = set()
    groups = defaultdict(list)
    for row in rows:
        if type(row["success"]) is not bool:
            raise ValueError("Success must be a JSON boolean")
        group = (row["suite"], int(row["task_id"]))
        key = (*group, int(row["episode"]), row["seed"])
        if key in seen:
            raise ValueError(f"Duplicate rollout: {key}")
        seen.add(key)
        groups[group].append(row["success"])
    rates = sorted(sum(v) / len(v) for v in groups.values())
    n, successes = len(rows), sum(r["success"] for r in rows)
    p, z = successes / n, 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    tail_n = max(1, math.ceil(len(rates) * 0.2))
    return {
        "episodes": n, "successes": successes, "failures": n - successes,
        "success_rate": p, "groups": len(groups),
        "zero_success_groups": sum(rate == 0 for rate in rates),
        "worst_pose_success": min(rates),
        "cvar20_pose_success": sum(rates[:tail_n]) / tail_n,
        "episode_wilson95_descriptive": [max(0, center - half), min(1, center + half)],
        "unlabeled_failures": sum(not r["success"] and r.get("failure_type") in (None, "UNLABELED") for r in rows),
        "missing_initial_states": sum(r.get("initial_state") is None for r in rows),
        "missing_video_references": sum(not r.get("video") for r in rows),
    }


def verify_hashes(root, entries):
    root = Path(root).resolve()
    for entry in entries:
        path = (root / entry["archive"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path points outside archive")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["public_sha256"]:
            raise ValueError(f"Hash mismatch: {entry['archive']}")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    manifest = load_json(root / "results/analysis/export_manifest.json")
    verify_hashes(root, manifest["files"])
    result = {"status": "VERIFIED_SAVED_RECORDS_ONLY", "method_effect": None,
              "reset_only_causal_effect": None, "exported_files_verified": len(manifest["files"])}
    for name, n, s, g in [("id", 120, 86, 12), ("robotinit", 800, 36, 80)]:
        path = root / "results/baseline" / name
        rows = [json.loads(line) for line in (path / "eval_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        stats = summarize(rows)
        require((stats["episodes"], stats["successes"], stats["groups"]) == (n, s, g), f"Unexpected principal sample: {name}")
        old = load_json(path / "summary.json")
        for field in ["success_rate", "successes", "worst_pose_success", "cvar20_pose_success"]:
            require(math.isclose(stats[field], old[field], abs_tol=1e-12), f"Summary mismatch: {name}/{field}")
        require(old["n_episodes"] == len(rows), f"Episode count mismatch: {name}")
        for actual, expected in zip(stats["episode_wilson95_descriptive"], old["wilson95"], strict=True):
            require(math.isclose(actual, expected, abs_tol=1e-12), f"Wilson interval mismatch: {name}")
        expected_groups = (
            [(suite, task) for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"] for task in range(3)]
            if name == "id" else
            [(r["suite"], int(r["task_id"])) for r in load_json(root / "splits/audit_workshop_80.json")]
        )
        expected_keys = {(suite, task, ep, 42 + i * 10000 + ep)
                         for i, (suite, task) in enumerate(expected_groups) for ep in range(10)}
        actual_keys = {(r["suite"], int(r["task_id"]), int(r["episode"]), r["seed"]) for r in rows}
        require(actual_keys == expected_keys, f"Frozen group/seed schedule mismatch: {name}")
        result[name] = stats
    split_sets = {}
    for name in ["audit", "adapt_pool", "dev", "heldout_test"]:
        values = load_json(root / "splits" / f"{name}.json")
        groups = {(r["suite"], int(r["task_id"])) for r in values}
        require(len(groups) == len(values), f"Duplicate split groups: {name}")
        for previous, other in split_sets.items():
            require(not groups & other, f"Overlapping split labels: {name}/{previous}")
        split_sets[name] = groups
    for child, parent in [("audit_workshop_80", "audit"), ("adapt_pool_workshop_100", "adapt_pool"), ("heldout_workshop_80", "heldout_test")]:
        groups = {(r["suite"], int(r["task_id"])) for r in load_json(root / "splits" / f"{child}.json")}
        require(groups <= split_sets[parent], f"Workshop split outside parent: {child}")
    result["core_split_sizes"] = {name: len(values) for name, values in split_sets.items()}
    result["split_identity_scope"] = "Suite-qualified labels only; physical reset disjointness unverified."
    result["descriptive_gap_pp"] = 100 * (result["id"]["success_rate"] - result["robotinit"]["success_rate"])
    receipt = load_json(root / "results/historical/run_receipts/g2_data_contract_blocked.json")
    require(receipt["gpu_training_started"] is False, "Unexpected G2 training status")
    result["training_status"] = "NOT_STARTED_IN_RECORDED_G2_ATTEMPT"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write-summary", type=Path)
    args = parser.parse_args(argv)
    result = verify(args.root.resolve())
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.write_summary:
        # Exclusive creation prevents accidental overwrites of evidence or previous analysis.
        with args.write_summary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
