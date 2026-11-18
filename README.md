# Earthworm image classification

Code for segmenting earthworm photographs, training classifiers and reproducing the analyses in the paper. The classifiers predict developmental stage, genus and species from segmented images.

| Resource | Link |
|---|---|
| Photographs, segmentations and metadata | [10.5281/zenodo.24341462](https://doi.org/10.5281/zenodo.24341462) |
| Model weights | [10.5281/zenodo.24341741](https://doi.org/10.5281/zenodo.24341741) |
| Code | [Version 1.0.1](https://github.com/darsa-group/worm-species/releases/tag/v1.0.3) |

## Installation

Use Python 3.11 and a separate environment. For CPU use:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -e '.[segmentation]'
```

For GPU use, install PyTorch and torchvision for your CUDA version before installing this project. `requirements-cpu-lock.txt` lists a pinned CPU environment. Activate the environment before running the commands below.

## Data and weights

Download the files from the two Zenodo records. Verify each record's `SHA256SUMS`. Put `metadata.csv` in `data/earthworm-images/` and extract `images.zip` into the same folder. Extract `earthworm-models.zip` into `data/`.

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

The image dataset contains 9,356 original photographs, 9,295 segmented photographs and one global metadata table. The table links each photograph to its segmentation and records failed segmentations. Repeated views and paired-camera photographs share an individual worm ID. Keep those observations together when dividing the data.

The weights record contains the segmentation model and one RGB ConvNeXt-Base classifier trained with seed 2340. To check the downloaded files and original split assignments:

```bash
make release-check
```

To use data on another disk, pass `DATASET=/path/to/earthworm-images MODELS=/path/to/earthworm-models` to the relevant commands.

## Use the models

```bash
make predict
make score
make segment
```

`predict` classifies the segmented photographs. `score` averages their probabilities by individual worm, then calculates the scores. `segment` processes the original photographs and saves foreground images, masks aligned with the original image and masks aligned with the rotated crop.

Outputs are written under `work/`. Existing outputs are preserved. For another run, choose a new path, for example `make predict PREDICT_OUTPUT=work/predictions-2.csv`. Use `DEVICE=cuda:0` for a GPU. Add `LIMIT=3` to process three photographs.

To use your own photographs:

```bash
python scripts/release.py segment --input /path/to/original.jpg --output work/new-segmentation
python scripts/release.py predict --input /path/to/segmented.jpg --output work/new-predictions.csv
```

The classifier expects a segmented image. Either command also accepts a directory of images.

## Training and analysis

Edit `configs/train.yaml` for your device and available memory. `batch_size: null` keeps the original experiment settings. Prepare the data, inspect the training plan, then run the workflow:

```bash
make prepare-data DEVICE=cuda:0
make reproduce-plan DEVICE=cuda:0
make reproduce DEVICE=cuda:0
```

The full plan contains 2,640 classifier fits across the baseline, visual-information and biological-coverage experiments. Training runs sequentially, followed by inference, analysis and plotting. Outputs go under `work/new_training/`. A completed fit is skipped when its settings and saved files match the plan. The original image order, whole-worm partitions and biological-control selections are retained.

The supplied ConvNeXt checkpoint is one fitted model. Reproducing the comparisons requires training the remaining fits. See `docs/training.md` for individual stages and output files, and `docs/data_contract.md` for the input formats.

Retraining the segmentation model requires the original human-annotated YOLO data and training split, which are outside the image release. `scripts/train_segmentation.py` contains the training recipe. 

