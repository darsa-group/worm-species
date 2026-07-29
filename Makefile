.DEFAULT_GOAL := help

PYTHON ?= python
PIPELINE_CONFIG ?= dev/genome_ablation_pipeline.yaml
PIPELINE_MODE ?= dry-run
PAPER_RESULT ?= paper_result
SPLIT_ROOT ?= .
DATA_ROOT ?= ../petridish-worm-images
REPORT_STYLE ?= dev/paper_report_style.yaml

PAPER_TESTS := \
	tests.test_paper_ablation_pipeline \
	tests.test_condition_variant_cache \
	tests.test_cache_maintenance \
	tests.test_data_transforms \
	tests.test_config_validation \
	tests.test_models \
	tests.test_training_losses

.PHONY: help ablation-pipeline paper-report test

help: ## Show the paper-pipeline commands.
	@echo "Worm Species paper pipeline"
	@echo
	@echo "  make ablation-pipeline                    Render the complete pipeline."
	@echo "  make ablation-pipeline PIPELINE_MODE=submit"
	@echo "                                             Submit its dependency chain."
	@echo "  make paper-report                         Rebuild completed-run paper outputs."
	@echo "  make test                                 Run the focused paper-pipeline tests."
	@echo
	@echo "Dry-run is the default; scheduler submission is always explicit."

ablation-pipeline: ## Render or submit caches, fits, collection, and report jobs.
	PYTHONPATH=.:src $(PYTHON) scripts/run_ablation_pipeline.py \
		--pipeline "$(PIPELINE_CONFIG)" --mode "$(PIPELINE_MODE)"

paper-report: ## Rebuild tables and figures from completed runs only.
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=.:src $(PYTHON) \
		scripts/build_paper_results.py \
		--paper-result "$(PAPER_RESULT)" \
		--split-root "$(SPLIT_ROOT)" \
		--data-root "$(DATA_ROOT)" \
		--style "$(REPORT_STYLE)"

test: ## Run the retained paper-pipeline verification surface.
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=.:src $(PYTHON) -m unittest $(PAPER_TESTS)
