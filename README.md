# Reproducing the earthworm classification paper

This branch contains the computational code and configurations for the paper's visual-information, biological-coverage and camera-acquisition experiments. It is a separate reproduction branch; the original development branch, manuscripts and reference results are retained.

We acquired images with a Canon camera in 2025 and 2026, and with a USB camera in 2026. The 2026 cameras photographed the same worms, in different poses. Correction therefore pairs **worm-mean representations**, not individual photographs. The principal camera correction is **USB → Canon** (`webcam_to_gphoto2` in filenames). The classifier remains fixed except in the explicitly labelled supervised head baseline.

## Quick start

Activate a compatible Python environment. On the original workstation:

```bash
conda activate wormspecies
```

For a new environment:

```bash
conda env create -f environment.yml
conda activate wormspecies-paper
```

Edit the `paths` section of **[configs/paper.yaml](configs/paper.yaml)** to point to the dataset, frozen split CSVs, training results, saved predictions and features. Edit **[configs/genome.yaml](configs/genome.yaml)** for your GenomeDK account, partition and conda activation path. Nothing in this branch includes model weights or biological images.

```bash
make check test                       # Config validation and numerical tests
make plan                            # Freeze checkpoints, cohorts and calibration folds
make inference features              # Local inference from trained weights, then embeddings
make all                             # CSV/NPZ analysis, tables, diagnostic plots and checks
```

**`make all` never trains a neural network, launches Slurm or loads neural-network checkpoints.** It expects inference artifacts and completed matched-control prediction CSVs. Its ridge/CORAL/PCA fits are lightweight statistical calibration of saved features. It deliberately stops if required artifacts are missing or inconsistent; it does not substitute example values.

For one checkpoint or reduced local resources:

```bash
make inference INDEX=0 DEVICE=cuda:0
make features INDEX=0 DEVICE=cuda:0
make analysis STAGE=baseline
make status
make -n all                           # Display commands without execution
```

`PYTHON=/path/to/env/bin/python` and `CONFIG=configs/my_machine.yaml` are supported by every target. Copy `configs/paper.yaml` for a new machine; keep that file outside the tracked package if it contains personal paths. Batch size, worker count and CPU threads are runtime settings in that YAML.

## Training on GenomeDK

Run these commands **from your GenomeDK terminal**. They do not connect through SSH from the local workstation.

```bash
make training-plan                   # Render five historical experiment stages; no submission
make training-submit                 # Explicitly submit the reviewed Slurm pipeline
# After baseline artifacts are available:
make plan
make controls-plan controls-slurm-plan
make controls-submit                 # Explicitly submit the additional matched controls
```

The historical stages reproduce architecture baselines, original visual experiments, combined visual interventions and biological exclusions. The recovery-only `resolution_gapfill.yaml` is provided separately; its conditions are already included in the complete visual configuration and are not submitted a second time.

Matched controls use a new full-training reference, stage-specific removals and three within-species random-removal subsets. The full-training class weights and output vocabulary are fixed; validation and test worms are unchanged. The matched-control worker refuses execution outside a Slurm job. Do not change its batch size while claiming an exact replication of the published matched-control experiment.

Transfer the prediction artifacts and their completion receipts back, then run:

```bash
make controls-compile
make all
```

The CSV-only compiler needs `audit_complete.json`, configuration JSONs, label maps, `run_summary.json`, complete `test_predictions_best.csv` files and any log referenced by a receipt. It does not need `best_model.pt`. See [docs/data_contract.md](docs/data_contract.md) for directory layouts and relocation rules.

## Rebuilding the prepared dataset

This is optional when the processed dataset is already available. Raw acquisition archives and a separately distributed segmentation checkpoint are required.

```bash
python -m pip install -r requirements-segmentation.txt
make dataset-plan                     # Inventory only
make dataset-prepare                  # Copy/reuse images; preserve original sources
make dataset-segment                  # Segmentation inference, not network training
```

The segmentation command can return a non-zero status when failed images are recorded. Inspect the saved failure records; do not hide these failures or repeatedly reprocess them to obtain a different scored population. The end-to-end analysis explicitly retains failures as abstentions.

## Outputs and scientific scope

All outputs go to the configured, separate directories; data and model directories are read-only analysis inputs. Results include per-seed tables, whole-worm sampling intervals, per-class F1, confusion counts, segmentation coverage/abstentions, calibration folds, transfer conditions, calibration-size curves, rank sensitivity, image-view aggregation and descriptive PCA/UMAP coordinates. [docs/experiments.csv](docs/experiments.csv) maps each paper component to its commands and output files.

- Primary external species comparisons score the six shared resolved species while retaining all eight classifier outputs. Predictions outside the scored classes still count as errors.
- Individual prediction averages per-image **probabilities** before selecting a class. Original visual experiments retain only hard decisions, so they use majority vote with lexical tie breaking.
- Seed SD describes model-training variability. Worm-bootstrap intervals describe conditional sampling uncertainty; fitted models/corrections are not retrained within bootstrap resamples.
- Calibration uses five whole-worm folds. Calibration and test worms are disjoint; folds are pooled before F1 is calculated.
- Fixed-rank eight correction is the primary calibration-size analysis. Adaptive rank up to 32 is a separately stored sensitivity analysis.
- CORAL does not require biological labels or exact photographic pose matches. Paired ridge uses corresponding worm means. Mean shift, Procrustes and supervised head ridge are retained as comparison methods.
- Greyscale remains an **original-domain visual experiment**. External greyscale transfer and reverse camera correction are excluded.
- UMAP is descriptive. Its appearance does not establish quantitative alignment or classification recovery.

The figure command produces clear diagnostic plots directly from exported CSVs, not the manuscript's hand-arranged publication layouts. This branch does not bundle manuscripts, poster sources, capture web interfaces, unrelated taxonomy pipelines or generated results.

## Reproducibility and release status

The code is adapted from the frozen project implementation; the numerical scoring/calibration core is retained and tested. [docs/VALIDATION.md](docs/VALIDATION.md) records what was exercised locally. [docs/tested_environment.txt](docs/tested_environment.txt) records observed dependency versions; `requirements.txt` gives installation bounds, not a claim that every supported combination was tested. GPU and Slurm execution require the appropriate hardware/environment.

Dataset/checkpoint access, archive identifiers and reuse licences must be supplied separately. **No data DOI, public data licence or code licence is asserted by this branch.** Checkpoint loading uses the project's trusted PyTorch checkpoint format; only load the distributed research checkpoints from a trusted source.
