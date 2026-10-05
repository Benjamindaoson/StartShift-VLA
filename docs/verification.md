# Verification scope

The supported command is `python scripts/verify_archive.py`. It needs only Python 3.12+ and the retained files. It checks all 920 principal records, historical metric agreement, group/seed identity, split-label separation and every imported file listed in the public manifest. Removing nonessential operational artifacts does not remove any verifier input.

Archive tests cover duplicate rollouts, invalid boolean values, empty evidence, suite-local task IDs, modified bytes and paths escaping the repository. The six tests passed in the prepared archive; they are rerun for the core publication. Source lint, syntax compilation, CLI help and saved-record verification are also part of the publication checks.

## Explicit integration limitation

The full historical suite was attempted in the archive's project-local Python 3.12.10 environment: **41 passed, 2 skipped, 1 failed**. `tests/test_groupdro.py` and `tests/test_rise.py` skipped through their existing PyTorch import guards. `tests/test_libero_id.py` failed with `ModuleNotFoundError: libero`. This lightweight environment has no complete robotics stack. All those tests remain in the repository.

No failed check was disabled to claim a green full suite. Component tests, packaged CLI help and saved-record arithmetic do not establish simulator replay, learned method performance or GPU integration. The previous full archive also passed clean Git/ZIP and Linux saved-record checks; the current core tree is verified independently before pushing.

## GitHub checks

The `Archive checks (not robotics integration)` workflow installs the lightweight pinned dependencies in a project-local `.venv`, checks saved evidence, runs the six archive tests, and performs lint and syntax compilation. It performs no training or simulation and downloads no research data or weights.

The precise publication validation report is `results/analysis/release_checks.json`. The source-to-public hashes are in `results/analysis/export_manifest.json`. `requirements-archive.txt` describes the lightweight archive-check environment, not the original GPU run environment.
