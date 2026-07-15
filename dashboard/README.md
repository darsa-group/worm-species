# Experiment results dashboard

This dashboard provides a read-only browser for heterogeneous SLURM and local
single-task experiments. By default it combines `outputs_slurm/` with
`single_task/outputs/` when both exist. It discovers schemas from the lightweight
files that are present, so interrupted and older runs remain visible even when
some artifacts are missing.

Each root keeps an explicit source label. Runs from `outputs_slurm/` and
`single_task/outputs/` therefore remain distinguishable even if they have the
same experiment or run name. Additional roots can be supplied with repeated
`--source LABEL=PATH` arguments.

The indexer never opens checkpoint bodies and never writes in the result tree.
It records checkpoint paths and file metadata only. Directory symlinks are not
followed; known JSON, CSV, TSV, text, and file-symlink artifacts are read with
explicit size limits. Broken links and malformed files become warnings.

## Build or refresh the index

The index is SQLite and uses only Python's standard library. Its default path is
`.cache/worm-species-dashboard/index.sqlite3`, outside the scientific result
directories.

```bash
python -m dashboard.index \
  --source slurm=outputs_slurm \
  --source single_task=single_task/outputs
```

An index path inside the selected results root is rejected. Use `--cache` to put
it elsewhere, and `--json` for a machine-readable scan summary.

## Launch the dashboard

Streamlit is an optional dependency:

```bash
python -m pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py -- \
  --source slurm=outputs_slurm \
  --source single_task=single_task/outputs
```

The application includes source, experiment, architecture, task, condition,
status, epoch, learning-rate, weight-decay, batch-size, pretrained/frozen
backbone, class-weighting, task-loss-weight, hierarchy-loss, and W&B filters.
Canonical resolved-configuration facets additionally include experiment type,
training seed, preprocessing image size, normalisation, mixed precision,
augmentation enablement, horizontal- and vertical-flip probabilities, rotation
degrees, early stopping, colour retention, W&B mode, and each named experimental
condition parameter. For example, `sigma`, `grid_size`, `retention`, and channel
order remain separate parameters; they are not collapsed into an ambiguous
generic strength value. Historical keys such as `data.image_size` and legacy
top-level condition parameters remain readable through additive aliases.

It also includes configuration and override views, best-validation and test
metrics, training curves, per-task reports and confusion-matrix heatmaps,
cue-suppression tables, and the matched-condition versus fixed-RGB stress
comparison. A single-task `macro_f1` is explicitly labelled single-task
macro-F1; it is never presented as a multitask mean.

Condition-matrix experiments have a separate **Condition Matrix** tab; they are
never merged with cue-suppression results. The tab filters model, training
condition, test condition, relation (`matched`, `rgb_stress`, or
`cross_condition`), and task. It displays completion warnings, relation counts,
a train-by-test macro-F1 heatmap, and the selected classification report and
confusion matrix with their source paths. The experiment-level aggregate is
preferred when present. Otherwise the dashboard reads only already-indexed
per-run matrix CSVs, with explicit artifact and row bounds.

The scientific matrix field `evaluation_relation` and the logging field
`condition_relation` are intentionally kept distinct. In matrix tables,
original-trained/original-tested is a matched cell. In canonical logging it is
labelled `original`; transformed matched tests, RGB stress tests, and cross-
condition tests retain their corresponding explicit labels. The dashboard does
not rewrite either source schema.

## Prepare combined confusion matrices

Completed historical runs already contain confusion-matrix CSVs. The optional
preparation step validates and combines those lightweight files into a cache
outside the result trees; it does not rerun inference or open checkpoints.

```bash
python -m worm_species.results.derive \
  --source slurm=outputs_slurm \
  --source single_task=single_task/outputs \
  --cache .cache/worm-species-dashboard/derived \
  --render all
```

Use `--render selected --run RUN_UID` for a smaller image set, or `--render none`
to build summaries only. The dashboard automatically joins
`.cache/worm-species-dashboard/derived/manifest.json` by source label and stable
run ID. Missing or malformed matrices produce warnings rather than changing a
scientific result directory.

For condition-matrix runs, preparation groups saved reports and confusion
matrices by test condition. It records per-task macro-F1 and prepares a combined
all-task image for each available condition. A derived mean macro-F1 is emitted
only when every configured task is present; incomplete task sets receive a
warning and no mean. The original root-level combined-image fields remain
available for older dashboard caches and consumers.

Both the SQLite index and derived images live under `.cache/` by default. Cache
locations inside a selected scientific result root are rejected. Refreshing or
rebuilding a cache can remove stale cache records, but it never edits result
JSON/CSV files, checkpoints, logs, splits, or scheduler state. Checkpoint bodies
are never opened by discovery or preparation.

Statuses are inferred from `run_status.txt`, `failed_runs.csv`, terminal metric
files, and artifact timestamps. `possibly_active` means recent partial files; it
does not claim that SLURM still has a live job.

## Supported artifacts

Run-level schemas include `config.json`, `test_metrics.json`, `history.csv`,
`split_summary.json`, `run_summary.json`, `label_to_index*.json`, classification
reports, confusion matrices, `run_overrides.args`, `run_status.txt`, and
checkpoint paths. Cue-suppression runs can additionally expose
`cue_suppression/{test_condition_metrics,macro_f1_ratios,transform_summary}.csv`.
Condition-matrix runs can expose
`condition_matrix_evaluation/{manifest.json,condition_metrics.csv,task_metrics.csv}`
and condition-specific classification reports and confusion matrices.

Experiment-level schemas include sweep plans, failed-run tables, colour-ablation
summaries, condition manifests, aggregate cue tables, and
`matched_vs_rgb_stress_test.csv`. Matrix experiments can additionally expose
`condition_matrix_evaluations.csv`, `condition_matrix_task_metrics.csv`, and
`condition_matrix_collection_summary.json`. Unknown files are ignored rather
than treated as evidence of completion.

The stored `out_dir` in a historical `run_summary.json` may point to transient
cluster scratch. Navigation always uses the discovered result path instead.

## W&B and local metric aliases

Local JSON and CSV artifacts remain the dashboard's source of truth. The
dashboard does not contact W&B, require a W&B login, or merge remote history
into local records. It exposes canonical local metric identities such as
`test/original/species_macro_f1` alongside the historical parsed fields so the
same run can be compared with the canonical W&B vocabulary.

The canonical W&B adapter may retain historical remote aliases for compatibility
while also logging slash-separated configuration and metric names. The
dashboard normalises those concepts once from the resolved local configuration;
it does not count a legacy alias and its canonical replacement as two scientific
measurements. Training condition, test condition, and condition relation remain
separate identities throughout filtering and condition-matrix views.
