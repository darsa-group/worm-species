# Training and analysis use the same files

Training is optional. `make all` analyses completed results without training a neural network. The public runner works locally and does not submit jobs.

## Configure a fresh run

First run `make prepare-data` with the released photographs and global metadata. Use `CONFIG=work/reproduction.yaml` with the individual training targets below, or use `make reproduce-plan` and `make reproduce`. The generated inputs preserve the original image order and whole-worm partitions; original acquisition archives and mask files are not required. Then edit `configs/train.yaml`:

- `output_root`: a fresh folder, separate from every reference input and existing analysis output.
- `device`: `cuda:0` or `cpu`. An explicitly requested unavailable CUDA device causes an error.
- `workers`, `cpu_threads`: loader and CPU resources.
- `batch_size`: `null` preserves the original per-experiment batch size. Set a smaller value (for example 16) if memory is limited. This change is recorded and is not presented as an identical historical training run.
- `control_batch_size`: 16, matching the new biological-control reference and removals.
- `wandb`: offline by default. Use `mode: online` after configuring your own W&B login, or `enabled: false` for no tracking. Local artifacts remain the scientific record.

The five files in `configs/training/` specify architecture/seed sweeps, loss, optimiser, stopping, visual transforms and biological exclusions. Scientific condition definitions are checked against `configs/visual_conditions.yaml`, which is also used to rescore saved predictions. Image caches are disabled in this portable runner; transforms are applied by the same loader code. No new train/test split is sampled.

```bash
make train-plan CONFIG=work/reproduction.yaml
make train-status CONFIG=work/reproduction.yaml
# Exactly one new fit:
make train CONFIG=work/reproduction.yaml EXPERIMENT=baseline INDEX=0
# All fits, sequentially on the configured device:
make train-all CONFIG=work/reproduction.yaml
```

`PYTHON`, `CONFIG`, `TRAIN_CONFIG` and `DEVICE` can be overridden. Use the same settings when planning and running: device/resource settings are frozen into the plan. A different setting requires a new `output_root`. No training is performed by `train-plan` or `train-status`.

| Experiment | Planned fits | Purpose |
|---|---:|---|
| baseline | 90 | Three architectures × 30 seeds |
| visual_ablation | 840 | 28 visual conditions × 30 seeds |
| visual_interactions | 600 | 20 combined conditions × 30 seeds |
| adult_taxon_baseline | 30 | Reference for biological exclusions |
| adult_taxon_holdouts | 330 | 11 excluded groups × 30 seeds |
| matched_controls | 750 | Full reference, stage removals and three matched-random subsets × 30 seeds |

The 2,640 fits are a complete rerun, not a quick local analysis. The runner starts one fit at a time. Completed fits are skipped only when their receipt, config and required outputs agree. Failed/partial fits remain preserved and require a fresh root; the command does not silently resume or overwrite them. The default historical batch size may exceed the memory available on a desktop GPU.

## Continue from new training into analysis

Planning writes `<output_root>/analysis.yaml`. It points inference to the new baseline checkpoints, visual scoring to the new experiment runs, biological-control compilation to the new control receipts, and every subsequent output to its own folder.

```bash
# After all required fits finish, for the default output root:
make plan CONFIG=work/new_training/analysis.yaml
make inference CONFIG=work/new_training/analysis.yaml
make features CONFIG=work/new_training/analysis.yaml
make all CONFIG=work/new_training/analysis.yaml
```

Replace the path if you changed `output_root`. These commands require the complete baseline grid, prepared 2026 images, and all required visual/control artifacts. They deliberately do not combine an incomplete new training grid with old fits. Existing reference results are never replaced.

## Saved-file interface

Each standard fit produces `reference/runs/<stage>/fit_<index>/<run-name>/` containing `config.json`, `label_to_index_by_task.json`, `best_model.pt`, `run_summary.json` and `test_predictions_best.csv`. Its parent contains `run_status.txt` and `training_complete.json`. Best checkpoints are selected using validation loss, not external-camera performance.

Controls use `controls/training/runs/<condition>/seed_<seed>/<run-name>/`, with the same run files plus `audit_complete.json` and `fixed_class_weights.json` in the parent. Removal applies only to whole training worms; validation/test images are unchanged. The complete reference training cohort supplies individual-count class weights to every control fit, and the output vocabulary is checked before training.
