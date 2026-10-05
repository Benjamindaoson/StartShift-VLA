# Research question

Can a vision-language-action policy retain its manipulation ability when the robot begins from a different valid configuration than the reset conditions common in its demonstrations?

The original project distinguished three possible responses: restore a familiar starting state, improve demonstration coverage, or condition the model on the initial state. The archive preserves that distinction because the evidence never established that a model modification was needed or beneficial.

The implemented observation interface uses an 8-dimensional end-effector/gripper representation. It does not encode the full robot joint configuration. The action has seven dimensions in total: six end-effector pose deltas and one gripper command. Initial-state sensitivity is therefore not interchangeable with proven joint-space robustness or a demonstrated failure mechanism.

The project originally asked whether explicit initial-state conditioning could outperform the strongest simple baseline at matched adaptation budgets, without harming ID competence. That method question remains unanswered. The completed work supports a narrower question: how different was the recorded base policy's success across these selected ID and RobotInit evaluation sets?

The archived answer is a substantial descriptive gap. Unequal task composition, absent reset-state traces and a single policy prevent interpreting it as a controlled estimate of the reset intervention alone. See [findings](findings.md) and [limitations](limitations.md).
