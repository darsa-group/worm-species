# Experiment results dashboard

This dashboard provides a read-only browser for heterogeneous experiments under
`outputs_slurm/`. It discovers schemas from the lightweight files that are
present, so interrupted and older runs remain visible even when some artifacts
are missing.

The indexer never opens checkpoint bodies and never writes in the result tree.
It records checkpoint paths and file metadata only. Directory symlinks are not
followed; known JSON, CSV, TSV, text, and file-symlink artifacts are read with
explicit size limits. Broken links and malformed files become warnings.

## Build or refresh the index

The index is SQLite and uses only Python's standard library. Its default path is
`.cache/worm-species-dashboard/index.sqlite3`, outside the scientific result
directories.

```bash
python -m dashboard.index --results-root outputs_slurm
```

An index path inside the selected results root is rejected. Use `--cache` to put
it elsewhere, and `--json` for a machine-readable scan summary.

## Launch the dashboard

Streamlit is an optional dependency:

```bash
python -m pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py -- --results-root outputs_slurm
```

The application includes experiment/run filters, inferred completion status,
configuration and override views, best-validation and test metrics, training
curves, per-task reports and confusion matrices, cue-suppression tables, and the
matched-condition versus fixed-RGB stress comparison.

Statuses are inferred from `run_status.txt`, `failed_runs.csv`, terminal metric
files, and artifact timestamps. `possibly_active` means recent partial files; it
does not claim that SLURM still has a live job.

## Supported artifacts

Run-level schemas include `config.json`, `test_metrics.json`, `history.csv`,
`split_summary.json`, `run_summary.json`, `label_to_index*.json`, classification
reports, confusion matrices, `run_overrides.args`, `run_status.txt`, and
checkpoint paths. Cue-suppression runs can additionally expose
`cue_suppression/{test_condition_metrics,macro_f1_ratios,transform_summary}.csv`.

Experiment-level schemas include sweep plans, failed-run tables, colour-ablation
summaries, condition manifests, aggregate cue tables, and
`matched_vs_rgb_stress_test.csv`. Unknown files are ignored rather than treated
as evidence of completion.

The stored `out_dir` in a historical `run_summary.json` may point to transient
cluster scratch. Navigation always uses the discovered result path instead.
