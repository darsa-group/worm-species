# Task-specific multitask generalisation experiments

This experiment family tests two distinct explanations for weak
developmental-stage performance:

1. negative transfer from sharing features and optimisation with taxonomy; and
2. insufficient biological support or species–stage confounding in the
   development data.

The ordinary predefined split and every structured holdout preserve the
existing individual-level split contract. The joint sampler changes only how
training individuals are drawn; it never changes validation or test
composition.

All architecture modes use the existing condition-aware loader and evaluator,
so the same model configs remain compatible with original baselines,
visual-ablation input conditions, and structured cohort holdouts.

## Architectures

`shared_heads` is the default and retains the historical state-dict layout:

```text
Shared backbone
├── Genus head
├── Species head
└── Developmental-stage head
```

Configurations and checkpoints that omit `model.multitask_architecture`
continue to resolve to this architecture.

`single_task` uses the same backbone and data path but constructs only one
head:

```yaml
model:
  multitask_architecture: single_task
  target_task: age  # genus, species, or age
```

Only the selected target is loaded into the loss and metric loop. Comparing
the age-only run with `shared_heads` diagnoses whether joint optimisation is
associated with worse age performance. Failure in both models instead points
toward limited biological support rather than multitask transfer alone.

`split_taxonomy_age` separates the final representation:

```text
Input
  │
Shared early backbone
  ├── Taxonomy final stage ── Genus head
  │                        └─ Species head
  └── Age final stage ─────── Age head
```

For torchvision and timm-style ConvNeXt models, the stem and first three
stages are shared and the final stage is deep-copied. The two copies start
with equal weights but have no aliased parameters. Other backbones use:

```text
shared vector + taxonomy bottleneck adapter
shared vector + age bottleneck adapter
```

Use `branch_mode: auto` for this selection, or explicitly request
`duplicated_final_stage` or `residual_adapter`. Explicit final-stage
duplication fails early for an unsupported backbone.

## Pooling and auxiliary mechanisms

Global average pooling remains the default:

```yaml
model:
  pooling: {type: global_average}
```

Task attention creates separate taxonomy and age attention pools when the
backbone exposes a spatial map or token sequence:

```yaml
model:
  pooling: {type: task_attention, dropout: 0.1}
```

Vector-only backbones use their vector directly and log that attention pooling
is unavailable.

The optional age supervised-contrastive projection produces a normalised
128-dimensional age embedding. Positive pairs share stage; cross-species
positives are preferred, anchors without positives are skipped, and the valid
anchor count and proportion are reported.

The exploratory species adversary attaches a gradient-reversal species
classifier only to the age representation. Its reversal coefficient increases
linearly over `warmup_epochs`; taxonomy features are unaffected. Keep this
separate from the principal comparison because adversarial invariance may also
remove biologically useful information.

PCGrad projects only the early parameters shared by the task pathways.
Branch and head parameters retain ordinary task gradients. The standard
optimisation path is unchanged unless
`training.gradient_strategy.type: pcgrad` is selected.

## Joint species–stage sampling

```yaml
data:
  sampler:
    type: joint_species_stage
    replacement: true
    samples_per_epoch: null
```

For each draw, the sampler chooses an observed species × stage combination,
then an individual, then one image from that individual. Consequently, many
images of one worm do not give that worm extra group-level sampling mass.
`joint_species_stage_sampler.csv` records individual counts, image counts, and
the effective probability of every observed combination.

## Configurations

The diagnostic configurations are deliberately separate:

```text
configs/train/generalisation/
├── shared_heads.yaml
├── single_task_genus.yaml
├── single_task_species.yaml
├── single_task_age.yaml
├── split_taxonomy_age.yaml
├── split_joint_sampler.yaml
├── split_pcgrad.yaml
├── split_age_supcon.yaml
├── split_joint_sampler_pcgrad.yaml
├── split_full.yaml
└── split_species_adversary.yaml
```

The principal configurations run seeds 40, 41, and 42 over the original split
and four structured holdouts. The isolated, more expensive mechanism runs use
seed 42 unless promoted after inspection. `split_full` enables the split
architecture, joint sampler, PCGrad, and age contrastive loss. The species
adversary remains disabled there and has its own exploratory config.

Validate a matrix without training:

```bash
make generalisation-validate \
  GENERALISATION_CONFIG=configs/train/generalisation/split_full.yaml
```

Render an inspectable, resumable SLURM bundle without submission:

```bash
PYTHONPATH=.:src python -m worm_species.slurm launch \
  --config configs/train/generalisation/split_full.yaml \
  --cluster-config configs/clusters/genome.yaml \
  --override slurm.paths.results_root=outputs/generalisation \
  --artifacts-dir slurm_artifacts/generalisation/split_full \
  --dry-run
```

After inspecting that bundle, the corresponding live launch is the same
command with `--submit`. Architecture-specific run-name prefixes prevent
separate matrices from colliding below the shared results root. Rendering and
unit tests do not constitute real training or cluster validation.

## Outputs and reporting

Each applicable run writes:

```text
test_metrics_best.json
data_holdout_evaluation/task_metrics.csv
gradient_diagnostics.csv
joint_species_stage_sampler.csv
age_embeddings_best.npz
age_embeddings_best_metadata.csv
```

Optional files are written only when their mechanism is enabled. Rebuild the
completed-runs-only report with:

```bash
python -m worm_species.analysis.generalisation_report \
  --results-root outputs/generalisation \
  --output-dir outputs/generalisation_report
```

or:

```bash
make generalisation-report
```

The report recursively discovers runs with completed test metrics and writes
the requested raw tables, seed summaries, paired differences, generated
Markdown and LaTeX prose, and Figures A–E. Every scientific performance plot
shows seed observations and seed-level 95% intervals. Five or more seeds use a
deterministic seed bootstrap; smaller replicated sets use a seed-level
t-interval. Images, batches, and diagnostic steps are never treated as
independent replicates. PNG files use 300 dpi, and every available figure is
also exported as PDF and SVG.

## Interpretation

The original predefined random split mainly measures interpolation among
individuals drawn from the same observed biological support. Structured
holdouts ask whether a model generalises to a withheld species–stage
combination, juvenile genus, or unseen species.

An age-only improvement over `shared_heads` is consistent with negative
transfer, especially if genus–age or species–age gradient cosines are often
negative. Improvement from splitting the final stage is also consistent with
reduced interference. Neither result proves that negative transfer caused the
original failure.

Joint-sampler improvement suggests that unequal support across observed
species–stage combinations mattered. Contrastive improvement suggests the age
representation became more stage-aligned across species. No improvement means
the tested mechanism did not improve generalisation under these runs; it does
not establish that the mechanism is absent.

Figure E is descriptive PCA (`n_components=2`, randomized solver,
`random_state=2026`). Visual separation is not formal evidence of
deconfounding.
