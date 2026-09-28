from startshift.training.baselines import lerobot_train_args


def _args(method):
    return lerobot_train_args(
        method=method,
        base_policy="base",
        dataset_repo="dataset",
        episodes=[1, 2],
        output_dir="out",
        steps=10,
        batch_size=2,
        lr=1e-4,
        seed=7,
    )


def test_standard_and_expert_ft_are_not_identical():
    standard = _args("standard-ft")
    expert = _args("expert-ft")
    assert "--policy.train_expert_only=false" in standard
    assert "--policy.train_expert_only=true" in expert
    assert standard != expert


def test_lora_uses_official_top_level_peft_flags():
    args = _args("lora")
    assert "--peft.method_type=LORA" in args
    assert "--peft.r=64" in args
    assert "--peft.lora_alpha=64" in args
