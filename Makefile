.DEFAULT_GOAL := help

PYTHON ?= python
PIPELINE_CONFIG ?= dev/genome_ablation_pipeline.yaml
PIPELINE_MODE ?= dry-run
PAPER_RESULT ?= paper_result
SPLIT_ROOT ?= .
DATA_ROOT ?= ../petridish-worm-images
REPORT_STYLE ?= dev/paper_report_style.yaml
ADULT_TAXON_PIPELINE_CONFIG ?= dev/genome_adult_taxon_ablation_pipeline.yaml
ADULT_TAXON_PIPELINE_MODE ?= dry-run
ADULT_TAXON_RESULT ?= adult_taxon_ablation_result
GENERALISATION_CONFIG ?= configs/train/generalisation/shared_heads.yaml
GENERALISATION_RESULTS ?= outputs/generalisation
GENERALISATION_REPORT ?= outputs/generalisation_report
GENERALISATION_CLUSTER ?= configs/clusters/genome.yaml
PERFORMANCE_CONFIG ?= configs/train/performance/performance_full.yaml
PERFORMANCE_RESULTS ?= outputs/performance
PERFORMANCE_REPORT ?= outputs/performance_report
PERFORMANCE_CLUSTER ?= configs/clusters/genome.yaml
PERFORMANCE_ARTIFACTS ?= submissions/performance_full
PERFORMANCE_GENOME_SOURCE ?= /home/devd/worm-species/wormsource2
PERFORMANCE_GENOME_RESULTS ?= outputs_slurm
PERFORMANCE_GENOME_REPORT ?= outputs/performance_genome_report
LOCAL_SMOKE_ROOT ?= local_slurm_simulation

PAPER_TESTS := \
	tests.test_paper_ablation_pipeline \
	tests.test_adult_taxon_ablation_pipeline \
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

.PHONY: help ablation-pipeline paper-report adult-taxon-ablation-pipeline adult-taxon-report generalisation-validate generalisation-report performance-validate performance-genome-source-check performance-genome-dry-run performance-genome-submit performance-genome-report performance-report performance-local-smoke test

help: ## Show the paper-pipeline commands.
	@echo "Worm Species paper pipeline"
	@echo
	@echo "  make ablation-pipeline                    Render the complete pipeline."
	@echo "  make ablation-pipeline PIPELINE_MODE=submit"
	@echo "                                             Submit its dependency chain."
	@echo "  make paper-report                         Rebuild completed-run paper outputs."
	@echo "  make adult-taxon-ablation-pipeline        Dry-run the Adult combination pipeline."
	@echo "  make adult-taxon-ablation-pipeline ADULT_TAXON_PIPELINE_MODE=submit"
	@echo "                                             Submit its 270-fit dependency chain."
	@echo "  make adult-taxon-report                   Rebuild Adult ablation figures."
	@echo "  make generalisation-validate              Validate one diagnostic run matrix."
	@echo "  make generalisation-report                Build completed-run diagnostic outputs."
	@echo "  make performance-validate                 Validate the 3-backbone performance matrix."
	@echo "  make performance-genome-dry-run           Render the Genome jobs without submitting."
	@echo "  make performance-genome-submit            Validate and submit the Genome jobs."
	@echo "  make performance-genome-report            Report completed Genome jobs from outputs_slurm."
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

adult-taxon-ablation-pipeline: ## Render or submit exhaustive Adult combination ablations.
	PYTHONPATH=.:src $(PYTHON) scripts/run_ablation_pipeline.py \
		--pipeline "$(ADULT_TAXON_PIPELINE_CONFIG)" \
		--mode "$(ADULT_TAXON_PIPELINE_MODE)"

adult-taxon-report: ## Rebuild the Adult combination ablation figures and tables.
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=.:src $(PYTHON) \
		scripts/build_adult_taxon_ablation_results.py \
		--paper-result "$(ADULT_TAXON_RESULT)" \
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
		--cluster-config "$(PERFORMANCE_CLUSTER)" \
		--override slurm.paths.project_root="$(PERFORMANCE_GENOME_SOURCE)"

performance-genome-dry-run: performance-validate ## Render the Genome performance jobs without submitting.
	PYTHONPATH=.:src $(PYTHON) -m worm_species.slurm launch \
		--config "$(PERFORMANCE_CONFIG)" \
		--cluster-config "$(PERFORMANCE_CLUSTER)" \
		--override slurm.paths.project_root="$(PERFORMANCE_GENOME_SOURCE)" \
		--artifacts-dir "$(PERFORMANCE_ARTIFACTS)" \
		--dry-run

performance-genome-source-check: ## Require the Genome runtime checkout to match this submission checkout.
	@expected="$$(git rev-parse HEAD)"; \
	actual="$$(git -C "$(PERFORMANCE_GENOME_SOURCE)" rev-parse HEAD 2>/dev/null || true)"; \
	if [ "$$actual" != "$$expected" ]; then \
		echo "Genome source checkout mismatch: expected $$expected, got $${actual:-missing}" >&2; \
		echo "Update $(PERFORMANCE_GENOME_SOURCE) before submitting." >&2; \
		exit 1; \
	fi; \
	if [ -n "$$(git -C "$(PERFORMANCE_GENOME_SOURCE)" status --porcelain --untracked-files=no)" ]; then \
		echo "Genome source checkout has tracked modifications: $(PERFORMANCE_GENOME_SOURCE)" >&2; \
		exit 1; \
	fi; \
	echo "Genome source checkout matches $$expected"

performance-genome-submit: performance-validate performance-genome-source-check ## Validate and submit the Genome performance jobs.
	PYTHONPATH=.:src $(PYTHON) -m worm_species.slurm launch \
		--config "$(PERFORMANCE_CONFIG)" \
		--cluster-config "$(PERFORMANCE_CLUSTER)" \
		--override slurm.paths.project_root="$(PERFORMANCE_GENOME_SOURCE)" \
		--artifacts-dir "$(PERFORMANCE_ARTIFACTS)" \
		--submit

performance-genome-report: ## Aggregate completed Genome performance runs.
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=.:src $(PYTHON) \
		-m worm_species.analysis.generalisation_report \
		--results-root "$(PERFORMANCE_GENOME_RESULTS)" \
		--output-dir "$(PERFORMANCE_GENOME_REPORT)"

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
