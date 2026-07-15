# Tables

Generated notebook tables and reusable analysis caches belong in
workflow-specific subdirectories here. This includes CSV, JSON, text, NumPy, and
argument-summary artefacts produced by analysis, diagnostics, Grad-CAM, and UMAP
notebooks.

The notebooks create their destination directories when run. Existing tables in
historical output directories are retained in place; the migration changes only
where future notebook-generated tables are written. Inputs, checkpoints, run
configs, split files, and live `outputs_slurm/` runs remain at their established
paths and are not copied or modified by this layout.
