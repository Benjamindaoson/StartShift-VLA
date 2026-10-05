from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

# Canonical task order from the upstream LIBERO benchmark. LIBERO-Plus replaces
# these benchmark registrations with thousands of perturbation variants, so ID
# evaluation must explicitly restore the unperturbed task definitions.
VANILLA_LIBERO_TASKS: dict[str, tuple[str, ...]] = {
    "libero_spatial": (
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ),
    "libero_object": (
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
    ),
    "libero_goal": (
        "open_the_middle_drawer_of_the_cabinet",
        "put_the_bowl_on_the_stove",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "put_the_bowl_on_top_of_the_cabinet",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_cream_cheese_in_the_bowl",
        "turn_on_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_wine_bottle_on_the_rack",
    ),
    "libero_10": (
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    ),
}


class _VanillaLiberoSuite:
    def __init__(self, suite_name: str):
        from libero.libero import benchmark, get_libero_path

        self.name = suite_name
        self.tasks = []
        for task_name in VANILLA_LIBERO_TASKS[suite_name]:
            bddl_file = f"{task_name}.bddl"
            init_states_file = f"{task_name}.pruned_init"
            bddl_path = Path(get_libero_path("bddl_files")) / suite_name / bddl_file
            init_path = Path(get_libero_path("init_states")) / suite_name / init_states_file
            if not bddl_path.is_file() or not init_path.is_file():
                raise FileNotFoundError(
                    f"Canonical LIBERO task assets are incomplete for {suite_name}/{task_name}: "
                    f"bddl={bddl_path.is_file()}, init_states={init_path.is_file()}"
                )
            self.tasks.append(
                benchmark.Task(
                    name=task_name,
                    language=benchmark.grab_language_from_filename(suite_name, bddl_file),
                    problem="Libero",
                    problem_folder=suite_name,
                    bddl_file=bddl_file,
                    init_states_file=init_states_file,
                )
            )
        self.n_tasks = len(self.tasks)

    def get_num_tasks(self) -> int:
        return self.n_tasks

    def get_task_names(self) -> list[str]:
        return [task.name for task in self.tasks]

    def get_task(self, i: int) -> Any:
        return self.tasks[i]


def install_vanilla_libero_benchmarks() -> None:
    """Restore canonical LIBERO suites after importing the LIBERO-Plus fork."""
    from libero.libero import benchmark

    for suite_name in VANILLA_LIBERO_TASKS:
        benchmark.BENCHMARK_MAPPING[suite_name] = partial(_VanillaLiberoSuite, suite_name)
