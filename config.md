# Paper-pipeline configuration

The repository has one supported workflow:
[`dev/genome_ablation_pipeline.yaml`](dev/genome_ablation_pipeline.yaml).
It composes these files:

| File | Purpose |
|---|---|
| `configs/defaults/base.yaml` | canonical data, training, split, cache, and logging defaults |
| `config.yaml` | segmented RGB input and preprocessing defaults |
| `configs/clusters/genome.yaml` | Genome paths, resources, environment, and node-shared scratch |
| `dev/genome_ablation_baseline.yaml` | 90 original-image baseline fits |
| `dev/genome_visual_ablation.yaml` | 660 matched visual-ablation fits |
| `dev/genome_visual_interactions.yaml` | 600 Gaussian pairwise-interaction fits |
| `dev/genome_data_holdouts.yaml` | 120 biological holdout fits |
| `dev/paper_report_style.yaml` | editable paper-figure styling |

Configuration inheritance uses `extends`. Child mappings are merged
recursively; child scalar and list values replace their parents. The pipeline
passes the cluster profile separately, then applies paper-specific persistent
cache roots as runtime overrides. Each phase uses seeds 40, 41, and 42 and
crosses every fit with hierarchy-loss weights 0 and 0.2.

## Pipeline controls

`base_cache` creates and verifies one persistent deterministic preprocessing
cache before any GPU stage. `condition_cache` creates one array task per
cacheable condition and depends on the base-cache job. Baseline training
depends on the base cache; visual training depends on both baseline completion
and all condition-cache tasks; interactions and holdouts follow as separate
dependent phases.

Relevant pipeline keys:

| Key | Meaning |
|---|---|
| `dependency` | must be `afterok` |
| `base_cache.directory_name` | persistent shared base-cache directory |
| `condition_cache.directory_name` | persistent shared condition-cache directory |
| `condition_cache.transforms` | deterministic transforms selected for precomputation |
| `condition_cache.max_active` | maximum concurrent CPU cache builders |
| `stages` | ordered baseline, visual, interaction, and holdout configs |
| `report` | final completed-runs-only paper build |

## Image and condition caches

The base cache key schema includes:

- the resolved inherited configuration hash;
- metadata and image source stamps;
- preprocessing image size;
- image and mask column identity;
- `crop_to_foreground` and `crop_pad`.

Paper configs explicitly set `data.crop_to_foreground: false`, because the
input column already points at segmented images.

`cache.condition_variants` controls runtime use of deterministic tensors:

| Key | Paper value |
|---|---|
| `enabled` | `true` for visual ablations |
| `protocol_version` | `1` |
| `storage` | `torch_float32` |
| `root` | injected by the pipeline or node-local staging job |

Protocol 1 applies the deterministic condition to the resized base tensor
before live random train augmentation. Validation and test transforms remain
deterministic. Condition identity includes the canonical condition,
protocol/schema version, and Torch/Torchvision versions.

## Visual ablations

`dev/genome_visual_ablation.yaml` defines 22 conditions:

- one zero-colour saturation control;
- four seeded patch-shuffle grids;
- ten Gaussian-blur severities;
- seven resolution-loss levels.

Only Gaussian blur, patch shuffle, and resolution loss are precomputed.
Resolution loss uses percentages
`[0, 25, 50, 75, 87.5, 93.75, 100]`, giving 224-pixel intermediate dimensions
`[224, 168, 112, 56, 28, 14, 1]`. Matched testing uses the conditioned cache;
the configured original-image cross-condition evaluation uses the unconditioned
base cache.

`dev/genome_visual_interactions.yaml` defines 20 ordered compound conditions.
Gaussian blur at 25%, 50%, 75%, or 100% is applied first, then paired
separately with zero colour or each patch-shuffle grid. Resolution is excluded
from this interaction matrix and retained as a standalone three-way control.

## Commands

```bash
# Render only; no sbatch calls.
make ablation-pipeline

# Submit cache, training, collection, and report dependencies.
make ablation-pipeline PIPELINE_MODE=submit

# Rebuild completed-run paper outputs with editable styling.
make paper-report

# Run the retained focused verification surface.
make test
```

Generated pipeline artifacts are written under `paper_result/artifacts/`.
Every production deployment-equivalent action here is a SLURM submission, so
submission is never the default.
