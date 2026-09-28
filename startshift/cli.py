from __future__ import annotations

import argparse
import json


def _cmd_manifest(args):
    from startshift.data.classification import load_pose_records
    from startshift.utils.io import write_json

    records = load_pose_records(args.classification, category=args.category)
    write_json([record.to_dict() for record in records], args.output)
    print(f"Wrote {len(records)} normalized records to {args.output}")


def _cmd_splits(args):
    from startshift.data.classification import load_pose_records
    from startshift.data.splits import build_pose_splits, save_pose_splits

    records = load_pose_records(args.classification, category="robot")
    splits = build_pose_splits(
        records,
        seed=args.seed,
        audit_fraction=args.audit_fraction,
        dev_fraction=args.dev_fraction,
        adapt_pool_fraction=args.adapt_pool_fraction,
        adapt_budgets=tuple(args.adapt_budgets),
    )
    save_pose_splits(splits, args.output_dir)
    print(json.dumps({k: len(v) for k, v in splits.items()}, indent=2))


def _cmd_make_targeted(args):
    from startshift.data.splits import save_failure_targeted_splits

    summary = save_failure_targeted_splits(
        args.pool,
        args.records,
        args.output_dir,
        budgets=args.budgets,
    )
    print(json.dumps(summary, indent=2))


def _cmd_train(args):
    from startshift.training.train import train_rise

    train_rise(
        args.config,
        split=args.split,
        method=args.method,
        limit_per_task=args.limit_per_task,
        output_dir=args.output_dir,
        seed=args.seed,
    )


def _cmd_baseline(args):
    from startshift.config import load_config
    from startshift.training.baselines import run_baseline, shell_command

    cfg = load_config(args.config)
    command = run_baseline(
        method=args.method,
        base_policy=cfg.policy.base_policy,
        dataset_repo=cfg.data.dataset_repo,
        split_path=args.split,
        output_dir=args.output_dir,
        steps=args.steps or cfg.train.steps,
        batch_size=args.batch_size or cfg.train.batch_size,
        lr=args.lr or cfg.train.lr,
        limit_per_task=args.limit_per_task,
        seed=args.seed if args.seed is not None else cfg.train.seed,
        execute=args.execute,
    )
    if not args.execute:
        print(shell_command(command))


def _cmd_eval(args):
    from startshift.evaluation.runner import evaluate_split

    evaluate_split(
        policy_path=args.policy,
        split_path=args.split,
        output_dir=args.output_dir,
        episodes_per_task=args.episodes,
        seed=args.seed,
        device=args.device,
        method_name=args.method_name,
        hard_reset=not args.soft_reset,
        init_states=not args.disable_benchmark_init_states,
        record_trajectories=args.record_trajectories,
        metadata={
            "adaptation_budget": args.adaptation_budget,
            "trainable_parameters": args.trainable_parameters,
            "gpu_hours": args.gpu_hours,
            "benchmark_init_states_disabled": args.disable_benchmark_init_states,
            "soft_reset": args.soft_reset,
        },
    )


def _cmd_eval_id(args):
    from startshift.evaluation.runner import evaluate_id_suites

    evaluate_id_suites(
        policy_path=args.policy,
        suites=args.suites.split(","),
        output_dir=args.output_dir,
        tasks_per_suite=args.tasks_per_suite,
        episodes_per_task=args.episodes,
        seed=args.seed,
        device=args.device,
        method_name=args.method_name,
    )


def _cmd_failure_template(args):
    from startshift.evaluation.failures import write_annotation_template

    write_annotation_template(args.records, args.output)


def _cmd_apply_failures(args):
    from startshift.evaluation.failures import apply_annotations

    apply_annotations(args.records, args.annotations, args.output)


def _cmd_audit(args):
    from startshift.evaluation.audit import audit

    audit(args.robotinit, output_path=args.output, id_records=args.id)


def _cmd_aggregate(args):
    from startshift.analysis.aggregate import write_aggregate

    paths = write_aggregate(
        args.results_root,
        args.output_dir,
        baseline_method=args.baseline_method,
        method=args.method,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


def _cmd_report(args):
    from startshift.analysis.report import build_report

    path = build_report(args.results_root, args.output_dir)
    print(path)


def _cmd_external_eval(args):
    from startshift.evaluation.external import official_eval_args, run_official_eval

    commands = official_eval_args(
        policy_path=args.policy,
        split_path=args.split,
        output_dir=args.output_dir,
        episodes=args.episodes,
        batch_size=args.batch_size,
    )
    for command in run_official_eval(commands, execute=args.execute):
        print(command)


def _cmd_state_coverage(args):
    from lerobot.datasets import LeRobotDataset

    from startshift.diagnostics.state_coverage import write_pose_distance_csv

    dataset = LeRobotDataset(args.dataset_repo, download_videos=False)
    write_pose_distance_csv(dataset, args.output, reference_task=args.reference_task)


def _cmd_gate(args):
    from startshift.gates import run_gate

    result = run_gate(
        mode=args.mode,
        output=args.output,
        id_summary=args.id_summary,
        robotinit_summary=args.robotinit_summary,
        baseline_heldout=args.baseline_heldout,
        method_heldout=args.method_heldout,
        baseline_id=args.baseline_id,
        method_id=args.method_id,
    )
    print(json.dumps(result.to_dict(), indent=2))
    if args.fail_on_reject and not result.passed:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="startshift", description="StartShift-VLA experiment CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifest", help="Normalize LIBERO-Plus task classification metadata")
    p.add_argument("--classification")
    p.add_argument("--category", default="robot")
    p.add_argument("--output", default="splits/robotinit_manifest.json")
    p.set_defaults(func=_cmd_manifest)

    p = sub.add_parser("make-splits", help="Build leak-free RobotInit audit/adapt/dev/test splits")
    p.add_argument("--classification")
    p.add_argument("--output-dir", default="splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--audit-fraction", type=float, default=0.20)
    p.add_argument("--dev-fraction", type=float, default=0.20)
    p.add_argument("--adapt-pool-fraction", type=float, default=0.30)
    p.add_argument("--adapt-budgets", type=int, nargs="+", default=[10, 25, 50, 100])
    p.set_defaults(func=_cmd_splits)

    p = sub.add_parser(
        "make-targeted",
        help="Select failure-targeted adaptation groups using base-policy scores from the adaptation pool only",
    )
    p.add_argument("--pool", default="splits/adapt_pool.json")
    p.add_argument("--records", required=True)
    p.add_argument("--output-dir", default="splits")
    p.add_argument("--budgets", type=int, nargs="+", default=[10, 25, 50, 100])
    p.set_defaults(func=_cmd_make_targeted)

    p = sub.add_parser("train", help="Train RISE-E, RISE-EA or RISE-EAR")
    p.add_argument("--config", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--method", choices=["rise-e", "rise-ea", "rise-ear"])
    p.add_argument("--limit-per-task", type=int)
    p.add_argument("--output-dir")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("baseline", help="Run or print an official LeRobot baseline command")
    p.add_argument("--config", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--method", choices=["standard-ft", "expert-ft", "lora"], required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--limit-per-task", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=_cmd_baseline)

    p = sub.add_parser("eval", help="Evaluate a base/RISE policy on a RobotInit split")
    p.add_argument("--policy", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--method-name")
    p.add_argument("--adaptation-budget", type=int, default=0)
    p.add_argument("--trainable-parameters", type=int, default=0)
    p.add_argument("--gpu-hours", type=float)
    p.add_argument(
        "--disable-benchmark-init-states",
        action="store_true",
        help="Disable LIBERO-Plus benchmark init states. This is NOT a physical reset-controller baseline.",
    )
    p.add_argument("--soft-reset", action="store_true")
    p.add_argument("--record-trajectories", action="store_true")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("eval-id", help="Evaluate the same policy on vanilla LIBERO")
    p.add_argument("--policy", required=True)
    p.add_argument("--suites", default="libero_spatial,libero_object,libero_goal,libero_10")
    p.add_argument("--tasks-per-suite", type=int, default=10)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--method-name")
    p.set_defaults(func=_cmd_eval_id)

    p = sub.add_parser("external-eval", help="Use official lerobot-eval for another policy/backbone")
    p.add_argument("--policy", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=_cmd_external_eval)

    p = sub.add_parser("failure-template", help="Create a manual failure annotation CSV")
    p.add_argument("--records", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=_cmd_failure_template)

    p = sub.add_parser("apply-failures", help="Merge manual failure labels back into eval records")
    p.add_argument("--records", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--output")
    p.set_defaults(func=_cmd_apply_failures)

    p = sub.add_parser("audit", help="Aggregate ID/RobotInit/tail/failure metrics")
    p.add_argument("--robotinit", required=True)
    p.add_argument("--id")
    p.add_argument("--output", required=True)
    p.set_defaults(func=_cmd_audit)

    p = sub.add_parser("gate", help="Evaluate phenomenon or method success gates")
    p.add_argument("--mode", choices=["phenomenon", "method"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--id-summary")
    p.add_argument("--robotinit-summary")
    p.add_argument("--baseline-heldout")
    p.add_argument("--method-heldout")
    p.add_argument("--baseline-id")
    p.add_argument("--method-id")
    p.add_argument("--fail-on-reject", action="store_true")
    p.set_defaults(func=_cmd_gate)

    p = sub.add_parser("aggregate", help="Aggregate multi-seed results and optional matched comparisons")
    p.add_argument("--results-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--baseline-method")
    p.add_argument("--method")
    p.set_defaults(func=_cmd_aggregate)

    p = sub.add_parser("report", help="Build plots and Markdown report from experiment results")
    p.add_argument("--results-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("state-coverage", help="Extract first-frame proprioception and pose distances")
    p.add_argument("--dataset-repo", default="lerobot/libero_plus")
    p.add_argument("--output", default="results/pose_distances.csv")
    p.add_argument("--reference-task", type=int)
    p.set_defaults(func=_cmd_state_coverage)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
