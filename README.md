# Paper analysis

A smaller branch for analysing the earthworm paper's results and running local inference from existing models. Neural-network training and cluster setup are omitted. Frozen visual-scoring conditions are retained in `configs/visual_conditions.yaml`. The full reproduction package remains on `paper-oublish`.

## Run the analysis

```bash
conda env create -f environment.yml
conda activate wormspecies-paper
```

If you already have the working environment, use `conda activate wormspecies` instead.

Edit **[configs/paper.yaml](configs/paper.yaml)** to point to your data and saved results, then run:

```bash
make test
make all
```

`make all` reads saved prediction CSVs, feature NPZs and completed biological-control results. It calculates F1, bootstrap intervals and calibration results, then creates tables and diagnostic plots in the configured output folders. It does not run neural-network training or image inference. Ridge, CORAL and PCA are fitted to saved features.

Required inputs include the dataset manifest, original worm-level splits, inference/feature plans, baseline validation summaries, visual-experiment predictions and matched-control CSVs with their frozen configurations and completion receipts. **Images, model weights and result files are distributed separately.** See [docs/data_contract.md](docs/data_contract.md) for their layout.

## Optional local inference

If you have processed images and trained checkpoints, create the saved predictions/features first:

```bash
make plan
make inference
make features
make all
```

For one checkpoint or a CPU run:

```bash
make inference INDEX=0 DEVICE=cpu
make features INDEX=0 DEVICE=cpu
```

The config controls GPU device, batch size, CPU threads and loader workers. Every command accepts `PYTHON=/path/to/python` and `CONFIG=configs/my_machine.yaml`. Use `make status` to inspect inference completion and `make -n all` to preview analysis commands.

## What is analysed?

- Baseline and task-specific visual results, including the original greyscale experiment.
- Canon/USB camera comparisons using individual mean probabilities and shared species.
- USB → Canon feature calibration with five disjoint worm folds; fixed rank eight and separate adaptive-rank sensitivity.
- Matched biological-removal contrasts, segmentation failures and inference-view sensitivity.
- Supplementary tables, descriptive PCA/UMAP and CSV-derived plots.

All photographs of a worm stay together. Seed variability and worm-sampling uncertainty are reported separately. Original results remain inputs; new outputs use separate folders. [docs/experiments.csv](docs/experiments.csv) lists individual analyses. The plots are diagnostic figures, not the manuscript's arranged layouts.

Optional dataset preparation: install `requirements-segmentation.txt`, then use `make dataset-plan`, `make dataset-prepare` and `make dataset-segment`. Keep failed segmentation records for the end-to-end analysis.

[Validation notes](docs/VALIDATION.md) describe what was checked. Dataset/checkpoint access, licences and archive identifiers must be provided separately.
