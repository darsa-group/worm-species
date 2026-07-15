# Figures

Generated notebook figures belong in workflow-specific subdirectories. Source
notebooks and tracked scientific artefacts should not be placed here solely to
silence version-control status.

The migrated notebooks create their own subdirectories beneath this directory,
including colour-robustness, cue-suppression, sweep-analysis, diagnostics,
Grad-CAM, and UMAP workflows. Existing figures in historical output directories
are retained in place; the migration changes only the destination of future
notebook-generated figures.

Do not treat this directory as a checkpoint or experiment-result store. Canonical
run inputs and scientific outputs remain under their existing `outputs/` or
`outputs_slurm/` paths.
