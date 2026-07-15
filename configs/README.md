# Configuration files

Configuration is plain YAML with recursive, mapping-only inheritance through
`extends`. Lists are replaced, not concatenated. Dotted `key=value` overrides
remain supported and the resolved value is always an ordinary Python dictionary.

## Inheritance and precedence

```text
configs/defaults/base.yaml
            |
            v
       config.yaml                 safe one-run quick start
            |
            +-- configs/experiments/standard.yaml
            +-- configs/experiments/hierarchy.yaml
            +-- configs/experiments/dual_cue.yaml
            |        +-- ghpc_dual_cue.yaml
            +-- configs/experiments/colour_ablation.yaml
            |        +-- ghpc_colour_ablation.yaml
            +-- configs/experiments/patch_shuffle_matrix.yaml
            +-- configs/experiments/persistent_hierarchy.yaml
                         |
                         +-- persistent_hierarchy_wandb.yaml

configs/clusters/local.yaml        independent machine axis
configs/clusters/genome.yaml
            |
            +-- genome_persistent.yaml
configs/clusters/ghpc.yaml
```

For local training, the child YAML overrides its parent. For a SLURM plan, the
general precedence is:

```text
explicit CLI --override
    > explicitly imported --legacy-env compatibility values
    > cluster profile
    > experiment child
    > config.yaml
    > configs/defaults/base.yaml
```

Ambient historical launcher variables are ignored unless `--legacy-env` is
requested. Cluster files contain machine resources and paths; they must not
change scientific run-spec hashes.

## Which file to use

| Goal | Experiment configuration | Typical count |
| --- | --- | ---: |
| One local standard run | `config.yaml` | 1 |
| Two-model standard sweep | `configs/experiments/standard.yaml` | 2 |
| Standard sweep with hierarchy consistency | `configs/experiments/hierarchy.yaml` | 2 |
| Full matched cue and RGB-stress study | `configs/experiments/dual_cue.yaml` | 224 |
| GHPC dual-cue historical W&B settings | `configs/experiments/ghpc_dual_cue.yaml` | 224 |
| Four-model 2×2/4×4 patch matrix | `configs/experiments/patch_shuffle_matrix.yaml` | 12 trainings, 36 evaluation cells |
| Colour retention 100% through 0% | `configs/experiments/colour_ablation.yaml` | 202 |
| GHPC colour W&B settings | `configs/experiments/ghpc_colour_ablation.yaml` | 202 |
| Genome persistent-cache hierarchy sweep | `configs/experiments/persistent_hierarchy.yaml` | 2 |
| Same persistent sweep with W&B | `configs/experiments/persistent_hierarchy_wandb.yaml` | 2 |

Use `local.yaml` for rendering and CPU-only planning, `genome.yaml` for the
Genome dual-cue workflow, `genome_persistent.yaml` for Genome persistent-cache
sweeps, and `ghpc.yaml` for GHPC node-local scratch. GHPC requires an explicit
GPU-node list before rendering or submission.

## Preferred commands

Validate and inspect the safe root configuration without touching data:

```bash
PYTHONPATH=src python -m worm_species.config.validate \
  --config config.yaml --workflow training

PYTHONPATH=src python -m worm_species.config.inspect \
  --config config.yaml --workflow training --format yaml
```

Inspect an experiment and cluster plan:

```bash
make validate \
  CONFIG=configs/experiments/dual_cue.yaml \
  CLUSTER=configs/clusters/local.yaml

make inspect \
  CONFIG=configs/experiments/patch_shuffle_matrix.yaml \
  CLUSTER=configs/clusters/local.yaml
```

Render without submission, run one local process, or explicitly submit:

```bash
make dry-run \
  CONFIG=configs/experiments/dual_cue.yaml \
  CLUSTER=configs/clusters/genome.yaml \
  ARTIFACTS_DIR=slurm/generated/dual-cue-check

make train TRAIN_CONFIG=config.yaml

make submit \
  CONFIG=configs/experiments/dual_cue.yaml \
  CLUSTER=configs/clusters/genome.yaml
```

`make dry-run` never calls `sbatch`. Direct equivalents are available through
`PYTHONPATH=src python -m worm_species.slurm launch --dry-run ...` and
`--submit`. Scheduler submission is never implied by rendering.

Use `--check-paths` with the config validator only on a machine where the data
and predefined split paths should exist. Dry-run and cluster planning leave this
off so a login node does not need the full dataset tree.

See [the full configuration reference](../config.md) for every
important switch, condition semantics, resource field, and worked example.
