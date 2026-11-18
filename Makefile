PYTHON ?= python
CONFIG ?= configs/paper.yaml
INDEX ?=
DEVICE ?=
STAGE ?= all
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)
PAPER = $(PYTHON) scripts/paper.py --config $(CONFIG)
INDEX_ARG = $(if $(INDEX),--index $(INDEX),)
DEVICE_ARG = $(if $(DEVICE),--device $(DEVICE),)

.DEFAULT_GOAL := help
.PHONY: help check all plan inference features analysis adaptive supplement figures verify status controls-compile release-freeze-check

help:
	@printf '%s\n' 'make release-check   Check downloaded images and model weights' 'make predict         Classify the segmented photographs' 'make segment         Segment the original photographs' 'make score           Score predictions by individual worm' 'make prepare-data    Prepare the original study splits' 'make reproduce-plan  List the fits for a full rerun' 'make reproduce       Train models and generate study results' 'See README.md for installation and input folders'
check:
	$(PAPER) check
release-freeze-check:
	$(PYTHON) scripts/check_release_freeze.py --require-links --require-committed
# These stages share outputs, so run them in order even when make uses -j.
# Training and neural inference have separate commands below.
all:
	$(PAPER) analysis
	$(PAPER) adaptive
	$(PAPER) controls-compile
	$(PAPER) supplement
	$(PAPER) figures
	$(PAPER) verify
plan:
	$(PAPER) plan
inference:
	$(PAPER) inference $(INDEX_ARG) $(DEVICE_ARG)
features:
	$(PAPER) features $(INDEX_ARG) $(DEVICE_ARG)
analysis:
	$(PAPER) analysis --stage $(STAGE)
adaptive:
	$(PAPER) adaptive
supplement:
	$(PAPER) supplement
figures:
	$(PAPER) figures
verify:
	$(PAPER) verify
status:
	$(PAPER) status
controls-compile:
	$(PAPER) controls-compile

TRAIN_CONFIG ?= configs/train.yaml
EXPERIMENT ?= all
TRAIN = $(PYTHON) scripts/paper_training.py --config $(CONFIG) --training-config $(TRAIN_CONFIG)
.PHONY: train-plan train train-all train-status
train-plan:
	$(TRAIN) plan $(DEVICE_ARG)
train:
	$(TRAIN) run --experiment $(EXPERIMENT) $(INDEX_ARG) $(DEVICE_ARG)
train-all:
	$(TRAIN) run $(DEVICE_ARG)
train-status:
	$(TRAIN) status --experiment $(EXPERIMENT) $(DEVICE_ARG)

DATASET ?= data/earthworm-images
MODELS ?= data/earthworm-models
LIMIT ?= 9356
THREADS ?= 4
REPRO_CONFIG ?= work/reproduction.yaml
PREDICTIONS ?= work/predictions.csv
PREDICT_OUTPUT ?= work/predictions.csv
SEGMENT_OUTPUT ?= work/segmentation
SCORE_OUTPUT ?= work/scores.json
FIGURE_OUTPUT ?= work/publication
RUNTIME ?= work/inputs
RESULTS ?= work/results
RELEASE = $(PYTHON) scripts/release.py
RELEASE_DEVICE = $(if $(DEVICE),--device $(DEVICE),--device cpu)
.PHONY: release-check prepare-data predict segment score reproduce-plan reproduce publication-figures
release-check:
	$(RELEASE) check --dataset $(DATASET) --models $(MODELS)
prepare-data:
	$(RELEASE) prepare --dataset $(DATASET) --runtime $(RUNTIME) --work $(RESULTS) --config $(REPRO_CONFIG) $(RELEASE_DEVICE) --threads $(THREADS)
predict:
	$(RELEASE) predict --dataset $(DATASET) --models $(MODELS) --output $(PREDICT_OUTPUT) --limit $(LIMIT) $(RELEASE_DEVICE) --threads $(THREADS)
segment:
	$(RELEASE) segment --dataset $(DATASET) --models $(MODELS) --output $(SEGMENT_OUTPUT) --limit $(LIMIT) $(RELEASE_DEVICE) --threads $(THREADS)
score:
	$(RELEASE) evaluate --models $(MODELS) --predictions $(PREDICTIONS) --output $(SCORE_OUTPUT)
reproduce-plan:
	$(PYTHON) scripts/reproduce.py plan --config $(REPRO_CONFIG) --training-config $(TRAIN_CONFIG) $(DEVICE_ARG)
reproduce:
	$(PYTHON) scripts/reproduce.py run --config $(REPRO_CONFIG) --training-config $(TRAIN_CONFIG) $(DEVICE_ARG)
publication-figures:
	$(PYTHON) scripts/publication_figures.py --config $(CONFIG) --output $(FIGURE_OUTPUT)
