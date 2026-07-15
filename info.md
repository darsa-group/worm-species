# Repository map

This repository now has one canonical Python package, one canonical trainer,
one configuration-driven experiment planner, and one read-only result-discovery
layer. Historical implementations are archived under `legacy/`; notebooks are
active scientific analysis code and are not legacy.

## Start here

| Goal | Preferred entry point |
| --- | --- |
| Understand configuration | [`config.md`](config.md) |
| Choose an experiment/cluster file | [`configs/README.md`](configs/README.md) |
| Validate a plan | `make validate` |
| Inspect resolved choices and counts | `make inspect` |
| Render without submission | `make dry-run` |
| Train one local model | `make train` |
| Explicitly submit to SLURM | `make submit` |
| Check result/job state | `make status` |
| Re-collect existing results | `make collect` |
| Prepare dashboard summaries | `make dashboard-prepare` |
| Launch the dashboard | `make dashboard` |
| Run CPU-only tests | `make test` |

The preferred direct training command is:

```bash
PYTHONPATH=src python -m worm_species.training \
  --config config.yaml --single-run
```

The preferred direct planning/submission surface is:

```bash
PYTHONPATH=src python -m worm_species.slurm launch --dry-run \
  --config configs/experiments/standard.yaml \
  --cluster-config configs/clusters/local.yaml \
  --artifacts-dir slurm/generated/standard-check
```

Replace `--dry-run` with `--submit` only after inspecting the rendered plan.

## Active layout

| Path | Responsibility |
| --- | --- |
| [`src/worm_species/`](src/worm_species/) | Canonical package: config, data, models, training, evaluation, experiments, SLURM, cache, and results. |
| [`config.yaml`](config.yaml) | Safe one-run quick-start configuration. |
| [`configs/defaults/`](configs/defaults/) | Detailed shared defaults. |
| [`configs/experiments/`](configs/experiments/) | Scientific experiment choices and externally expanded sweeps. |
| [`configs/clusters/`](configs/clusters/) | Local, Genome, persistent-cache, and GHPC machine settings. |
| [`slurm/templates/`](slurm/templates/) | Canonical rendered-job templates. |
| `slurm/generated/` | Ignored dry-run/submission plans; never scientific outputs. |
| [`dashboard/`](dashboard/) | Read-only multi-root result browser and index. |
| [`tests/`](tests/) | CPU-only configuration, scientific-contract, SLURM, dashboard, and compatibility tests. |
| [`notebooks/`](notebooks/) | Active analysis, data, diagnostics, and interpretability notebooks. |
| [`scripts/`](scripts/) | Active maintenance and transfer utilities. |
| [`docs/refactor/`](docs/refactor/) | Audit, migration, compatibility, and behaviour-contract records. |
| [`legacy/`](legacy/) | Archived historical implementations and opt-in restoration data. |

The canonical trainer implementation is under
[`src/worm_species/training/`](src/worm_species/training/). `train.py` is a thin
root convenience entry point. A submitted run specification always invokes the
canonical trainer once and fits one model; sweeps are expanded by the planner,
not inside the trainer.

## Experiments and results

Scientific results stay in place:

- `outputs_slurm/` contains multitask/SLURM result trees and must be treated as
  live, read-only input during discovery and dashboard indexing.
- `single_task/outputs/` contains local or historical single-task result trees.
- `split_csv/` contains externally linked predefined splits; do not rewrite or
  relocate the CSV links.
- `figures/` and `tables/` contain generated presentation artefacts.
- `logs/` contains generated logs, including scheduler logs where configured.
- checkpoints, datasets, W&B artefacts, and caches are never repository-audit
  inputs and are not moved by layout maintenance.

The dashboard combines both result roots without loading checkpoint bodies:

```bash
make dashboard-prepare \
  SLURM_RESULTS_ROOT=outputs_slurm \
  SINGLE_TASK_RESULTS_ROOT=single_task/outputs

make dashboard \
  SLURM_RESULTS_ROOT=outputs_slurm \
  SINGLE_TASK_RESULTS_ROOT=single_task/outputs
```

Prepared confusion-matrix images, per-task macro-F1 summaries, and the dashboard
index are cached under `.cache/worm-species-dashboard/`, outside scientific
result directories. Missing, partial, old, malformed, and still-running result
schemas become warnings rather than writes or crashes. See
[`dashboard/README.md`](dashboard/README.md) for detected fields and status
semantics.

## Notebooks

Notebooks are grouped by purpose:

- [`notebooks/analysis/`](notebooks/analysis/) for result comparison and cue or
  colour studies;
- [`notebooks/data/`](notebooks/data/) for dataset and split exploration;
- [`notebooks/diagnostics/`](notebooks/diagnostics/) for leakage and prediction
  diagnostics;
- [`notebooks/interpretability/`](notebooks/interpretability/) for CAM, Grad-CAM,
  and UMAP work.

They should import canonical `worm_species` modules and save new presentation
outputs beneath `figures/` and `tables/`. Notebook-relative paths should be
resolved from the repository root so execution from a notebook directory does
not redirect results into the source tree.

## Legacy archive and restore

Historical training scripts, old `src.*` adapters, prior run-spec generators,
and duplicated SLURM launcher bodies are archived rather than kept on the active
command surface. The archive does not include notebooks, splits, live results,
datasets, checkpoints, figures, tables, or active utilities.

Inspect the restoration manifest and destination paths first:

```bash
legacy/restore_compatibility.sh --dry-run
```

Restore the archived public paths only when reproducing an old workflow:

```bash
legacy/restore_compatibility.sh
```

The restore command preflights the complete manifest, is idempotent for
byte-identical files, and refuses to overwrite different content. It has no
force mode. See [`legacy/README.md`](legacy/README.md) and
[`legacy/compatibility.map`](legacy/compatibility.map) for exact historical to
canonical mappings.

## Current safety boundaries

- Experiment YAML controls scientific choices; cluster YAML controls resources,
  scratch, environment, and machine paths.
- Matched-condition training remains separate from original-RGB transformed-test
  stress evaluation.
- Condition matrices add evaluations, not training runs. The four-model patch
  example makes 12 fits and 36 train/test condition cells.
- Nested internal and external sweep expansion is rejected before submission.
- Dry runs do not call `sbatch`.
- Result discovery, collection, derivation, and dashboard indexing do not modify
  `outputs_slurm/` or `single_task/outputs/`.
- New source code belongs under `src/worm_species/`; do not revive duplicated
  full implementations at historical root paths.
