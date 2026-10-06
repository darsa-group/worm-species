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
.PHONY: help check test all plan inference features analysis adaptive supplement figures verify status controls-compile

help:
	@printf '%s\n' 'make test  - Check configuration and numerical tests' 'make all   - Analyse saved results and build tables/plots' 'Optional inference: make plan inference features' 'Overrides: PYTHON=... CONFIG=... DEVICE=cpu INDEX=0 STAGE=baseline'
check:
	$(PAPER) check
test: check
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'
# Sequential recipes deliberately prevent make -j from racing shared output stages.
# This target never plans/submits/runs neural training and never performs inference.
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

.PHONY: dataset-plan dataset-prepare dataset-segment
dataset-plan:
	$(PAPER) dataset-plan
dataset-prepare:
	$(PAPER) dataset-prepare
dataset-segment:
	$(PAPER) dataset-segment $(DEVICE_ARG)
