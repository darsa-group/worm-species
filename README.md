# Earthworm image classification

Code for the earthworm image study, including segmentation, classifier training, evaluation and the analyses used in the paper. The companion Zenodo DOIs are [10.5281/zenodo.24341462](https://doi.org/10.5281/zenodo.24341462) for the photographs and [10.5281/zenodo.24341741](https://doi.org/10.5281/zenodo.24341741) for the model weights. The code release is planned for [v1.0.0](https://github.com/darsa-group/worm-species/releases/tag/v1.0.0).

## Installation

Use Python 3.11 in a new environment. For a CPU installation:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -e '.[segmentation]'
```

For GPU use, install PyTorch and torchvision for your CUDA version before installing this project. `requirements-cpu-lock.txt` records the versions used during local preparation.

## Data and weights

Put `metadata.csv` in `data/earthworm-images/` and extract `images.zip` there. Extract `earthworm-models.zip` into `data/`. Verify each record's `SHA256SUMS` before extraction. The resulting folders should be:

```text
data/
  earthworm-images/
    metadata.csv
    2025/
    2026/
  earthworm-models/
    classification/
    segmentation/
```

There are 9,356 original photographs and 9,295 segmented photographs. Calibration photographs and processing intermediates are excluded. One global metadata table links each original image to its segmentation. The model record contains the segmentation checkpoint and an RGB ConvNeXt-Base classifier trained with seed 2340.

```bash
make release-check
```

This checks the image paths, original split assignments and model hashes. To keep the files on another disk, pass `DATASET=/path/to/earthworm-images MODELS=/path/to/earthworm-models` to the commands below.

## Run the supplied models

```bash
make predict
make score
make segment
```

`predict` classifies the released segmented photographs. `score` averages their probabilities by individual worm before calculating scores. `segment` processes the original photographs and saves the foreground images and both mask coordinate systems. Each command writes to `work/` and preserves existing outputs. Choose another output path for a new run, for example `make predict PREDICT_OUTPUT=work/predictions-2.csv`.

Use `DEVICE=cuda:0` for a GPU. Add `LIMIT=3` to `predict` or `segment` for a short run. For your own photographs:

```bash
python scripts/release.py predict --input /path/to/segmented.jpg --output work/new-predictions.csv
python scripts/release.py segment --input /path/to/original.jpg --output work/new-segmentation
```

The classifier expects a segmented photograph. The segmentation command accepts an original photograph. A directory can be used in place of a file.

## Reproduce the study

Edit `configs/train.yaml` for your device and available memory. A null batch size keeps the original experiment settings. Prepare the original splits, inspect the training plan, then run the workflow:

```bash
make prepare-data DEVICE=cuda:0
make reproduce-plan DEVICE=cuda:0
make reproduce DEVICE=cuda:0
```

The full plan contains 2,640 classifier fits across baseline, visual-information and biological-coverage experiments. The runner trains one fit at a time, then runs inference, analysis and plotting. Outputs go under `work/new_training/`. Completed fits are skipped only when their settings and saved files agree. Original worm identities, split order and biological-control selections are retained. See `docs/training.md` for individual stages and output files.

The supplied classifier is one representative fit. Reproducing all comparisons requires training the omitted fits. The code and saved-result calculations were checked locally; the full training grid was not rerun during release preparation. Retraining the segmentation model requires its original human-annotated YOLO data and split, which are outside this image release. `scripts/train_segmentation.py` provides the recorded recipe. Generated masks do not replace those annotations.

## Repository and licence

The earlier research commands remain available through `make -f Makefile.research help` in the Git repository. The earlier README is in `docs/RESEARCH_WORKSPACE.md`. Public helper scripts use the `release_` prefix so those research workflows can remain available. The source archive contains the public workflow, with no images, weights, generated results or test suite.

Original code uses MIT. Keep the third-party notices in `THIRD_PARTY_NOTICES.md`, including Ultralytics AGPL-3.0, torchvision BSD and Tabler MIT terms. Data and model licences are stated in the Zenodo records. Cite the versioned code release and both companion records. For the release commit and tag, see `docs/RELEASE_FREEZE.md`.
