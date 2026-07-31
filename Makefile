.DEFAULT_GOAL := help

PYTHON ?= python
PIPELINE_CONFIG ?= dev/genome_ablation_pipeline.yaml
PIPELINE_MODE ?= submit
PAPER_RESULT ?= paper_result
SPLIT_ROOT ?= .
DATA_ROOT ?= ../petridish-worm-images
REPORT_STYLE ?= dev/paper_report_style.yaml
GENERALISATION_CONFIG ?= configs/train/generalisation/shared_heads.yaml
GENERALISATION_RESULTS ?= outputs/generalisation
GENERALISATION_REPORT ?= outputs/generalisation_report
GENERALISATION_CLUSTER ?= configs/clusters/genome.yaml
PERFORMANCE_CONFIG ?= configs/train/performance/performance_full.yaml
PERFORMANCE_RESULTS ?= outputs/performance
PERFORMANCE_REPORT ?= outputs/performance_report
LOCAL_SMOKE_ROOT ?= local_slurm_simulation

PAPER_TESTS := \
	tests.test_paper_ablation_pipeline \
	tests.test_condition_variant_cache \
	tests.test_cache_maintenance \
	tests.test_data_transforms \
	tests.test_config_validation \
	tests.test_models \
	tests.test_training_losses \
	tests.test_generalisation_configs \
	tests.test_generalisation_models \
	tests.test_generalisation_training \
	tests.test_generalisation_report \
	tests.test_performance_features

.PHONY: help ablation-pipeline paper-report generalisation-validate generalisation-report performance-validate performance-report performance-local-smoke test

help: ## Show the paper-pipeline commands.
	@echo "Worm Species paper pipeline"
	@echo
	@echo "  make ablation-pipeline                    Render the complete pipeline."
	@echo "  make ablation-pipeline PIPELINE_MODE=submit"
	@echo "                                             Submit its dependency chain."
	@echo "  make paper-report                         Rebuild completed-run paper outputs."
	@echo "  make generalisation-validate              Validate one diagnostic run matrix."
	@echo "  make generalisation-report                Build completed-run diagnostic outputs."
	@echo "  make performance-validate                 Validate the 3-backbone performance matrix."
	@echo "  make performance-report                   Build individual-level performance outputs."
	@echo "  make performance-local-smoke              Render and run a synthetic 5-epoch local simulation."
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

generalisation-validate: ## Validate one task-specific generalisation matrix.
	PYTHONPATH=.:src $(PYTHON) -m worm_species.slurm validate \
		--config "$(GENERALISATION_CONFIG)" \
		--cluster-config "$(GENERALISATION_CLUSTER)"

generalisation-report: ## Aggregate completed task-specific generalisation runs.
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=.:src $(PYTHON) \
		-m worm_species.analysis.generalisation_report \
		--results-root "$(GENERALISATION_RESULTS)" \
		--output-dir "$(GENERALISATION_REPORT)"

performance-validate: ## Validate one 3-backbone performance matrix.
	PYTHONPATH=.:src $(PYTHON) -m worm_species.slurm validate \
		--config "$(PERFORMANCE_CONFIG)" \
		--cluster-config "$(GENERALISATION_CLUSTER)"

performance-report: ## Aggregate completed performance runs.
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=.:src $(PYTHON) \
		-m worm_species.analysis.generalisation_report \
		--results-root "$(PERFORMANCE_RESULTS)" \
		--output-dir "$(PERFORMANCE_REPORT)"

performance-local-smoke: ## Render SLURM artifacts and run 5 local synthetic epochs.
	PYTHONPATH=.:src $(PYTHON) scripts/run_local_performance_smoke.py \
		--simulation-root "$(LOCAL_SMOKE_ROOT)"

test: ## Run the retained paper-pipeline verification surface.
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=.:src $(PYTHON) -m unittest $(PAPER_TESTS)
