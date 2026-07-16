# Development experiment file map

This document describes the files involved in the `devconfig` Genome workflow,
what each file contains, when it is used, and what is safe to change. It covers
configuration, planning, training, evaluation, collection, and dashboard files;
it is not an inventory of unrelated notebooks or historical archive files.

## Execution flow

```text
devconfig*.yaml + configs/clusters/genome.yaml
                    |
                    v
       configuration loading and validation
                    |
                    v
          generic sweep expansion
                    |
                    v
       one run specification per model fit
                    |
                    v
           rendered SLURM array job
                    |
                    v
       python -m worm_species.training
                    |
                    v
     outputs, collection, and dashboard index
```

The scientific invariant is:

```text
one generated run specification
    = one SLURM array task
    = one canonical trainer invocation
    = one model fit
```

Evaluation conditions reuse trained checkpoints and do not create additional
fits.

## Files you normally edit

### `devconfig.yaml`

Purpose: base matched-condition Genome experiment.

Current contents:

- ConvNeXt Base and ViT-B/16;
- learning rates `0.0005` and `0.0001`;
- early stopping enabled and disabled as a sweep dimension;
- eight Cartesian genus/species/age loss-weight combinations;
- hierarchy loss fixed off;
- 114 matched training conditions;
- evaluation catalogue retained but disabled;
- W&B online logging;
- output directory `output_allrun`.

Current size:

```text
64 parameter combinations x 114 training conditions = 7,296 fits
```

Safe changes:

- `training.epochs`, `batch_size`, `weight_decay`, AMP, workers, and validation
  interval;
- model and learning-rate lists;
- individual task-weight lists;
- preprocessing image size and train-only augmentation settings;
- W&B project/group/mode;
- complete entries under `sweep.conditions`;
- `output.out_dir`.

Important constraints:

- keep `multi_task.hierarchy_loss.enabled: false` in this base;
- keep `evaluation.test_conditions.enabled: false` for matched training;
- every `sweep.parameters` entry must be a non-empty list;
- each new condition adds 64 fits with the current parameter grid;
- adding independent lists creates a Cartesian product.

### `devconfig_hierarchy.yaml`

Purpose: matched-condition hierarchy-loss comparison.

It inherits all models, learning rates, task-weight choices, conditions, and
disabled evaluation from `devconfig.yaml`. It then:

- fixes hierarchy loss on;
- sweeps meaningful hierarchy weights `0.2`, `0.5`, and `1.0`;
- uses a separate W&B group and experiment identity.

Current size:

```text
192 parameter combinations x 114 training conditions = 21,888 fits
```

Safe changes:

- hierarchy weight values;
- W&B group;
- experiment identity.

Do not add `hierarchy_loss.enabled: [true, false]` here. That would recreate
disabled runs once for every irrelevant hierarchy weight.

### `devconfig_rgb_stress.yaml`

Purpose: original-RGB training followed by transformed-test stress evaluation.

It inherits the 114-condition evaluation catalogue from `devconfig.yaml`, but
replaces the training-condition list with only `original` and enables the test
schedule.

Current size:

```text
64 original-RGB fits
64 checkpoints x 114 evaluation conditions = 7,296 evaluation cells
```

Safe changes:

- inherited model, LR, task-weight, and early-stopping parameters through the
  base config;
- the evaluation condition catalogue in `devconfig.yaml`;
- W&B group and experiment identity.

Important constraint: keep `sweep.conditions` restricted to `original`. If a
transformed training condition is added while fixed-RGB evaluation is enabled,
planning fails because that checkpoint is not original-RGB trained.

### `config.yaml`

Purpose: safe one-run repository root and parent of the development configs.

It contains common defaults for data paths, preprocessing, augmentation,
training, task loss, W&B, cache, output, the resolved input condition, and
disabled sweep/evaluation blocks. Its lower half is a comment-only catalogue of
optional child-config examples.

Safe changes: machine-independent defaults that genuinely should affect every
child config.

Prefer changing a `devconfig*.yaml` child for an experiment-specific decision.
Changing `config.yaml` can alter every experiment that extends it.

### `configs/defaults/base.yaml`

Purpose: detailed inherited defaults and compatibility values.

It contains metadata columns, label mappings, taxonomic uncertainty handling,
split settings, cache settings, early-stopping defaults, task configuration,
and historical aliases accepted during migration.

Normally do not edit this for one experiment. A change here can affect the root
config and every child. Use it only for a repository-wide default decision.

### `configs/clusters/genome.yaml`

Purpose: Genome-specific scheduling and filesystem configuration.

It contains:

- account and GPU partitions;
- CPUs, memory, GPU count, and time limit;
- array concurrency (`slurm.array.max_active`);
- job-local cache/scratch behavior;
- conda activation path and environment name;
- project, data, metadata, results, cache, and log paths;
- setup, collection, cleanup, and monitoring resources.

Safe changes:

- actual Genome paths;
- partition/account values;
- time, memory, CPU, and concurrency requests;
- conda installation and environment;
- logging and monitoring choices.

Do not place model, task, cue, label, or loss choices in this file. Cluster
configuration must remain independent of scientific configuration.

### `Makefile`

Purpose: short user-facing orchestration commands.

Relevant targets:

```bash
make validate
make inspect
make dry-run
make submit
make status
make collect
make dashboard-prepare
make dashboard
```

Safe changes: thin command aliases and harmless default paths.

Do not put scientific sweep logic or complex scheduling behavior in Make
recipes. That logic belongs in validated Python modules.

## Configuration implementation files

These files are used automatically. Experiment users normally should not edit
them.

| File | Responsibility | Change only when |
| --- | --- | --- |
| `src/worm_species/config/loading.py` | Loads YAML, resolves `extends`, and recursively merges mappings. Lists replace parent lists. | Changing inheritance semantics repository-wide. |
| `src/worm_species/config/normalization.py` | Converts canonical ranges and accepted historical aliases into one canonical dictionary. | Adding a canonical configuration representation or migration. |
| `src/worm_species/config/ranges.py` | Expands endpoint-aware decimal ranges and formats condition names. | Adding generic range behavior. |
| `src/worm_species/config/sweeps.py` | Computes the parameter Cartesian product and combines it with complete conditions. | Changing generic sweep semantics. |
| `src/worm_species/config/schema.py` | Registry of public key paths, expected types, defaults, consumers, and status. | Adding or deprecating a public configuration key. |
| `src/worm_species/config/validation.py` | Type, range, transform, contradiction, and nested-expansion validation. | Adding a new rule or supported scientific option. |
| `src/worm_species/config/validate.py` | `python -m worm_species.config.validate` command. | Changing validator CLI presentation. |
| `src/worm_species/config/inspect.py` | Prints resolved choices and expected counts without training. | Adding inspection fields. |
| `src/worm_species/config/overrides.py` | Parses and applies dotted `key=value` overrides. | Changing override syntax. |
| `src/worm_species/config/migrate.py` | Converts historical config shapes into canonical YAML. | Adding a compatibility migration. |
| `src/worm_species/config/__main__.py` | Dispatches config subcommands. | Adding a supported config CLI command. |

The unfinished `src/worm_species/config/tui.py` working-tree draft is not part
of the validated workflow described here. It should not be relied upon until
its interrupted tests are completed and it is deliberately committed.

## SLURM implementation files

| File | Responsibility | Change only when |
| --- | --- | --- |
| `src/worm_species/slurm/config.py` | Merges experiment and cluster configs and validates resource/scratch settings. | Supporting a new cluster/resource rule. |
| `src/worm_species/slurm/planning.py` | Generates unique run specs, resolves one condition per task, prevents nested expansion, and checks result collisions. | Changing submission-plan semantics. |
| `src/worm_species/slurm/rendering.py` | Renders scripts and self-contained dry-run artifacts. | Changing template inputs or generated plan layout. |
| `src/worm_species/slurm/submission.py` | Calls `sbatch` and records dependency/job identifiers. | Changing actual submission behavior. |
| `src/worm_species/slurm/status.py` | Combines filesystem state with `squeue`/`sacct` where available. | Adding status sources or interpretations. |
| `src/worm_species/slurm/collection.py` | Re-runs result aggregation without training. | Adding an output schema or collector. |
| `src/worm_species/slurm/environment.py` | Resolves declared legacy environment/path inputs. | Maintaining an explicitly supported environment mapping. |
| `src/worm_species/slurm/cli.py` | Validate, inspect, launch, status, and collect command interface. | Adding a supported SLURM command. |
| `src/worm_species/slurm/__main__.py` | Enables `python -m worm_species.slurm`. | Changing module dispatch. |

### `slurm/templates/*.tmpl`

Purpose: shell templates rendered into setup, array, collector, and cleanup
jobs. The Genome workflow normally uses the job-local array and result
collector templates.

Do not edit templates to change an experiment. Edit them only when runtime
shell behavior, resource monitoring, scratch handling, or dependency execution
must change for all relevant submissions. Rendered copies belong under
`slurm/generated/` and are ignored generated state.

## Canonical trainer files

| File | Responsibility | Change only when |
| --- | --- | --- |
| `src/worm_species/training/cli.py` | Loads one resolved config, applies overrides, supports dry run, and invokes the runner. | Changing trainer CLI behavior. |
| `src/worm_species/training/modes.py` | Resolves explicit masked-label, hierarchy, W&B, condition, and evaluation switches. | Adding a training mode without another full script. |
| `src/worm_species/training/runner.py` | Coordinates metadata, splits, loaders, model, optimizer, epochs, checkpoints, evaluation, and result writes. | Changing the training lifecycle. |
| `src/worm_species/training/loaders.py` | Creates train/validation/test datasets and split-specific transforms. | Changing loader construction. |
| `src/worm_species/training/losses.py` | Missing-label masking, task weights, and hierarchy consistency loss. | Changing loss mathematics. |
| `src/worm_species/training/metrics.py` | Task and aggregate metrics, including empty-task behavior. | Changing metric definitions. |
| `src/worm_species/training/epochs.py` | CPU/GPU epoch loops and AMP behavior. | Changing optimization steps. |
| `src/worm_species/training/checkpoints.py` | Saves/loads established checkpoint schema and selects the best model. | Deliberately migrating checkpoint contracts. |
| `src/worm_species/training/reproducibility.py` | Applies fixed random seeds. | Changing reproducibility policy. |
| `src/worm_species/training/naming.py` | Run/output naming. | Changing result path contracts. |
| `src/worm_species/training/__main__.py` | Enables `python -m worm_species.training`. | Changing module dispatch. |

The SLURM array must always call the canonical trainer, never choose among old
training scripts.

## Data and transform files

| File | Responsibility | Change only when |
| --- | --- | --- |
| `src/worm_species/data/metadata.py` | Prepares multitask metadata without changing split membership. | Changing metadata policy. |
| `src/worm_species/data/taxonomy.py` | Genus/species parsing, uncertainty, and explicit overrides. | Changing taxonomy rules. |
| `src/worm_species/data/labels.py` | Stable label maps and missing/rare label handling. | Changing label contracts. |
| `src/worm_species/data/datasets.py` | Canonical multitask dataset output. | Changing sample structure. |
| `src/worm_species/data/image_validation.py` | Validates image files and modes. | Adding validation behavior. |
| `src/worm_species/data/cropping.py` | Mask-derived foreground cropping. | Changing crop mathematics. |
| `src/worm_species/data/transforms.py` | Composes resize, train-only augmentation, condition, and normalization in the preserved order. | Changing pixel behavior deliberately. |
| `src/worm_species/data/conditions.py` | Implements saturation, grayscale, channel shuffle, bilateral, Gaussian, and patch transforms. | Adding or changing a scientific cue. |

For an experiment, change condition objects in YAML rather than editing these
implementations.

## Evaluation files

| File | Responsibility | Change only when |
| --- | --- | --- |
| `src/worm_species/evaluation/cue_suppression.py` | Builds fixed-RGB transformed-test conditions and writes cue reports. | Changing stress-test behavior or schema. |
| `src/worm_species/evaluation/condition_matrix.py` | Evaluates selected trained checkpoints across named test conditions. | Changing cross-condition evaluation. |

Evaluation must remain separate from training expansion. A test condition is
not another fit.

## Generated planning files

A dry run writes a directory beneath `slurm/generated/`, containing files such
as:

| Generated file | Contents | May be changed manually? |
| --- | --- | --- |
| `resolved_config.yaml` | Fully merged experiment and cluster configuration. | No; regenerate it. |
| `submission_plan.json` | Counts, hashes, resources, paths, commands, and dependency graph. | No; regenerate it. |
| `sweep_plan.tsv` | One row per generated run specification. | No; regenerate it. |
| `run_specs/` | One resolved configuration/argument set per array task. | No; regenerate it. |
| rendered `.sh`/`.sbatch` files | Setup, array, collection, and cleanup commands. | No; fix config/code/template and regenerate. |

Dry-run artifacts are not scientific results and contain no datasets or
checkpoints.

## Scientific output files

The trainer and evaluators may create these files beneath the configured result
root:

| File | Contents |
| --- | --- |
| `config.json` | Resolved run configuration. |
| `run_overrides.args` | Exact externally assigned overrides. |
| `history.csv` | Epoch-level train and validation history. |
| `best_model.pt` | Best checkpoint under the established schema. |
| `test_metrics.json` | Final test metrics. |
| `classification_report_*.csv` | Per-task classification reports. |
| `confusion_matrix_*.csv` | Per-task confusion matrices. |
| `split_summary.json` | Recorded split counts/diagnostics. |
| `run_summary.json` | Run identity, status, best epoch, and summary fields. |
| condition/cue CSVs | Fixed-RGB, matched, or matrix evaluation results where enabled. |

These are scientific outputs. Do not edit them to change an experiment; change
the source YAML and submit a distinct run.

## Collection and dashboard files

| File | Responsibility |
| --- | --- |
| `src/worm_species/results/discovery.py` | Read-only discovery of SLURM and single-task result schemas. |
| `src/worm_species/results/derive.py` | Builds cached per-task metrics and combined confusion-matrix figures outside result trees. |
| `src/worm_species/results/normalization.py` | Normalizes old/new result metadata into dashboard facets. |
| `dashboard/app.py` | Streamlit entry point. |
| `dashboard/index.py` | External SQLite cache keyed by path, size, and modification time. |
| `dashboard/views.py` | Filters, summaries, curves, reports, and comparisons. |
| `dashboard/condition_matrix.py` | Train-condition by test-condition views. |
| `dashboard/README.md` | Launch commands and detected result fields. |

Dashboard caches belong under `.cache/worm-species-dashboard/`. Dashboard and
collection code must not modify `outputs_slurm/` or load checkpoint bodies just
to populate an index.

## Files that are inputs, not configuration

- `split_csv/*.csv` are externally linked predefined split memberships. Do not
  move, rewrite, or regenerate them during experiment setup.
- datasets and metadata trees are external inputs selected through paths.
- checkpoints and existing `outputs_slurm/` runs are scientific records.
- W&B artifacts are outputs; local `config.json`, metrics, histories, and
  reports remain the source of truth.

## Commands for these files

Matched training:

```bash
make validate CONFIG=devconfig.yaml CLUSTER=configs/clusters/genome.yaml
make inspect CONFIG=devconfig.yaml CLUSTER=configs/clusters/genome.yaml
make dry-run CONFIG=devconfig.yaml CLUSTER=configs/clusters/genome.yaml
```

Hierarchy training:

```bash
make validate CONFIG=devconfig_hierarchy.yaml CLUSTER=configs/clusters/genome.yaml
make dry-run CONFIG=devconfig_hierarchy.yaml CLUSTER=configs/clusters/genome.yaml
```

Original-RGB stress evaluation:

```bash
make validate CONFIG=devconfig_rgb_stress.yaml CLUSTER=configs/clusters/genome.yaml
make dry-run CONFIG=devconfig_rgb_stress.yaml CLUSTER=configs/clusters/genome.yaml
```

Use `make submit` only after checking the exact counts and rendered plan from
the corresponding dry run.

For configuration syntax and ranges, see `config.md`. For experiment examples,
see `devconfig.md`. For the broader repository map, see `info.md`.
