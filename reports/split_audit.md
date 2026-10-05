# Split audit

- Seed: `42`.
- Core groups: audit `310`, adapt_pool `465`, dev `310`, heldout_test `465`.
- Every core split covers all `20/20` suite × difficulty strata.
- Pairwise core overlap: `0` for every pair.
- Workshop M0 RobotInit subset: `80` groups (`4` per stratum).
- Workshop targeted-selection pool: `100` groups (`5` per stratum).
- Workshop held-out subset: `80` groups (`4` per stratum).
- Clean ID subset: first `3` benchmark tasks per suite (`12` tasks).
- Random and difficulty adaptation budgets are nested and remain within adapt_pool.

Machine-readable distributions, overlap arrays, checks, and SHA-256 receipts are in `reports/split_audit.json`.
