"""Project-wide keys and defaults."""

OBS_STATE = "observation.state"
ACTION = "action"
EPISODE_INDEX = "episode_index"
FRAME_INDEX = "frame_index"
TASK_INDEX = "task_index"

STARTSHIFT_INITIAL_STATE = "startshift.initial_state"
STARTSHIFT_RELATIVE_STATE = "startshift.relative_state"
STARTSHIFT_POSE_ID = "startshift.pose_id"
STARTSHIFT_DIFFICULTY = "startshift.difficulty"
STARTSHIFT_GROUP_INDEX = "startshift.group_index"

ROBOTINIT_CATEGORY_ALIASES = {
    "robot",
    "robot_init",
    "robot-initial-state",
    "robot_initial_state",
    "robot initial state",
    "robot initial states",
    "initial_state",
    "initial pose",
    "initial_pose",
}

DEFAULT_SEED = 42
DEFAULT_ADAPT_BUDGETS = (10, 25, 50, 100)
