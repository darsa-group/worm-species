# Final refactor report

## Outcome

The repository received a bounded, behavior-preserving structural refactor.
The audit used tracked files as its primary manifest and a pruned search for
untracked source-relevant files. Generated results, datasets, caches,
checkpoints, model weights, W&B artifacts, and SLURM outputs were excluded from
content traversal. No legacy entry point or import path was removed.

The refactor centralizes configuration handling, sweep expansion, model
construction, label/split helpers, dual-cue run specification generation, and
dual-cue result collection under `src/worm_species/`. Historical root scripts
and `src.*` modules remain compatibility wrappers or re-export surfaces.

## Audit deliverables

- `PRE_REFACTOR_REPORT.md`: bounded source, entry-point, notebook,
  configuration-key, shell/SLURM, dynamic-path, and proposed-refactor
  inventories;
- `repository_inventory.json` and `repository_inventory.md`: the pre-refactor
  75-file manifest and excluded-directory summary;
- `../post_refactor/repository_inventory.json` and `.md`: the
  post-refactor 94-file source-relevant manifest;
- `FILE_MIGRATION_MAP.md`: old-to-canonical path and compatibility map;
- `NOTEBOOK_COMPATIBILITY.md`: notebook parsing, imports, and path assumptions;
- `BEHAVIOUR_CONTRACT_RESULTS.md`: automated and baseline differential results.

## Preserved contracts

- All seven public Python CLI `--help` surfaces are byte-for-byte unchanged.
- Baseline and refactored dual-cue generators both produce 224 identical run
  specification files from `config.yaml`.
- Existing trainer module names, root entry-point paths, importable public
  helpers, option names, option defaults, and positional arguments remain.
- Model head shapes and state-dict parameter names remain.
- Ordinary, colour-ablation, and cue-suppression checkpoint top-level schemas
  remain.
- Predefined split CSV path and read semantics remain.
- Training-only sorted class-map construction and missing-label behavior remain.
- Matched-condition training stays separate from fixed-RGB transformed-test
  evaluation.
- Generated child runs disable internal sweep expansion, preventing nested
  Cartesian sweeps.
- Root shell and SLURM launchers were not structurally refactored.
- Notebook files, cells, imports, outputs, and execution assumptions were not
  changed.

## Validation

`PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v` passes all
15 contract tests. The bounded 38-file Python source set compiles, all root
shell scripts pass `bash -n`, and `git diff --check` passes. No tracked diff is
present below `outputs_slurm` or an `outputs` directory.

## Deliberately unchanged

The refactor does not merge single-task and multi-task datasets, does not unify
the scientific training/evaluation variants, and does not rewrite SLURM,
scratch, cache, notebook, checkpoint, or W&B workflows. Those areas lack the
real-data, checkpoint, GPU, or cluster fixtures needed to demonstrate exact
equivalence. Existing unrelated behavior and bugs were not silently changed.

## Remaining validation boundary

No claim is made for full-dataset training, GPU/AMP numerical equivalence,
real-checkpoint evaluation, W&B network behavior, clean-kernel notebook
execution, or live SLURM scheduling/scratch/copy-back behavior. Those require
the original data, hardware, services, and cluster environment.

## Concurrent repository state

The working tree was already dirty before this task, including changes in
`.gitignore`, shared `src` files, two node-local launchers, and
`tranfser_from_genome.sh`; these changes were preserved. Live untracked run
directories continued to appear under `outputs_slurm` and were not inspected
or modified.

During final validation, the three tracked files under `split_csv/` changed
from regular files to symbolic links targeting an external SSD. No refactor
command wrote those paths. They remain untouched because restoring or accepting
that change could alter the scientific split and requires an explicit owner
decision.

No refactor commit was created because committing would mix the repository's
pre-existing and concurrent user-owned changes with the scoped refactor.
