from pathlib import Path

from startshift.evaluation.libero_id import (
    VANILLA_LIBERO_TASKS,
    install_vanilla_libero_benchmarks,
)


def test_vanilla_libero_suite_overlay_restores_canonical_tasks() -> None:
    from libero.libero import benchmark, get_libero_path

    install_vanilla_libero_benchmarks()

    init_root = Path(get_libero_path("init_states"))
    bddl_root = Path(get_libero_path("bddl_files"))
    for suite_name, expected_names in VANILLA_LIBERO_TASKS.items():
        suite = benchmark.get_benchmark_dict()[suite_name]()
        assert suite.get_task_names() == list(expected_names)
        assert suite.get_num_tasks() == 10
        for task in suite.tasks:
            assert (bddl_root / task.problem_folder / task.bddl_file).is_file()
            assert (init_root / task.problem_folder / task.init_states_file).is_file()
