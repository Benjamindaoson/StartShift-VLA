# Current findings

The principal baseline records contain 86 successes in 120 ID episodes (71.67%) and 36 in 800 RobotInit episodes (4.50%). The difference is 67.17 percentage points. This is strong descriptive evidence that the selected RobotInit setting was substantially harder for this SmolVLA checkpoint than the selected ID setting.

| RobotInit suite | Successes / episodes | Success rate |
|---|---:|---:|
| Spatial | 21 / 200 | 10.50% |
| Object | 0 / 200 | 0.00% |
| Goal | 6 / 200 | 3.00% |
| Long (`libero_10`) | 9 / 200 | 4.50% |

Sixty-five of the 80 RobotInit groups recorded zero successes in ten trials. Worst-group success and lower-tail CVaR over the worst 20% of groups are both zero. Difficulty levels 1–5 have 10, 12, 5, 5 and 4 successes respectively out of 160 trials each. This ordering is not strictly monotonic; it should not be narrated as a validated difficulty-response curve.

The historical episode-level Wilson 95% intervals are 63.03–78.96% for ID and 3.27–6.17% for RobotInit. Repeated episodes within a task/pose group are clustered, and task selection was restricted. These intervals describe the recorded samples under a simplifying episode-level model; they are not confidence intervals for a causal reset effect or population-wide VLA reliability.

The observations are consistent with initialization sensitivity. They do not isolate training-reset dependence as the cause: ID includes 12 canonical tasks while RobotInit includes 80 variants with different task composition. Saved trajectories, actual initial state vectors, paired scene identities and failure annotations are unavailable.

There is no observed RISE improvement, degradation or comparison against a trained matched-data baseline. The effect estimate is **N/A**, not zero. The project closes with an empirical warning and a documented evidence gap, rather than a validated mitigation method.
