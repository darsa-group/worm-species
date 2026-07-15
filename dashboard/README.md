# Experiment results dashboard

This dashboard provides a read-only browser for heterogeneous SLURM and local
single-task experiments. By default it combines `outputs_slurm/` with
`single_task/outputs/` when both exist. It discovers schemas from the lightweight
files that are present, so interrupted and older runs remain visible even when
some artifacts are missing.

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
