# Core archive provenance

This public tree is a minimal publication of the preserved StartShift-VLA research archive. It is committed on top of the existing GitHub `main` history; no force push or private operational Git history is needed. Earlier public designs remain available in Git history. The original working tree, full preparation archive and migration material remain backed up privately.

The research source was based on commit `1e7ab0648f46c5c2ac4ce23a998630bac5b38b90` and included local modifications and untracked evidence. That commit alone does not identify the exported source. `results/analysis/export_manifest.json` records each retained imported artifact's original hash, public hash, relative source name and transformation.

## Retained evidence

- The complete 120-row ID and 800-row RobotInit principal evaluation JSONL, with their historical summaries. Episode outcomes, seeds, step counts, rewards and failure labels are preserved.
- Four frozen parent partitions, three workshop subsets, the RobotInit manifest, the ID selection and `adapt_random_25.json` referenced by the G2 receipt. The retained split JSON and principal JSONL are byte-identical to the private source.
- The failed initial ID launch log and its exit receipt, the successful ID/RobotInit run start/end receipts, the descriptive audit/G1 gate, and the G2 receipt stating that training did not begin.
- Core source, original tests, configuration recipes, historical scripts, split-audit reports and upstream revisions.

## Intentionally omitted

Raw datasets, videos, weights, checkpoints, GPU telemetry CSVs, repetitive terminal progress logs, preflight retries, duplicate adaptation-budget lists, the large generated architecture viewer, duplicate old README and internal preparation/review documents. None is needed for the supported principal-result verifier. These omissions do not erase the incomplete state of the research: the code audit and limitations explicitly describe it.

The public Git history preserves earlier public code and documentation. Private backups are not pushed. SSH credentials, local migration notes, host/container identifiers, machine-local paths, virtual environments, caches and temporary files remain excluded.

## Relocation and sanitization

The principal output directories become `results/baseline/id/` and `results/baseline/robotinit/`. The essential logs and receipts retain their relative subtree under `results/historical/`.

Public log/summary copies replace local paths and host/IP identifiers with neutral placeholders and remove terminal ANSI sequences. The local policy path is replaced by its official model ID and recorded revision. Historical shell scripts have archive warnings; the ID resume script takes its local policy path from an environment variable. Python and shell source line endings are normalized to LF for portable Git hashes. No model structure or scientific algorithm was changed for this publication.

Generic hardware/software versions are retained as reproducibility context. Original copyright attribution is retained. Hash checks establish consistency of the retained artifacts against this manifest, not a third-party signature or proof of simulator execution.

## Reproduction boundary

`python scripts/verify_archive.py` validates retained evidence offline. It does not rerun simulation, establish missing data provenance or validate RISE. No new experiments, training or research-data downloads were performed for publication.
