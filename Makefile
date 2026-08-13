.DEFAULT_GOAL := help

PYTHON ?= python
PIPELINE_CONFIG ?= dev/genome_ablation_pipeline.yaml
PIPELINE_MODE ?= dry-run
PAPER_RESULT ?= paper_result
SPLIT_ROOT ?= .
DATA_ROOT ?= ../petridish-worm-images
REPORT_STYLE ?= dev/paper_report_style.yaml
HOLDOUT_VISUAL_RESULT ?= paper_result/notebook_holdout_visual_figures
HOLDOUT_VISUAL_MODEL ?= convnext_base
SPECIES_ABLATION ?= Aporrectodea_longa
ADULT_TAXON_PIPELINE_CONFIG ?= dev/genome_adult_taxon_ablation_pipeline.yaml
ADULT_TAXON_PIPELINE_MODE ?= dry-run
ADULT_TAXON_RESULT ?= adult_taxon_ablation_result
PUBLICATION_PIPELINE_CONFIG ?= dev/genome_publication_30seed_pipeline.yaml
PUBLICATION_PIPELINE_MODE ?= dry-run
PUBLICATION_RESULT ?= publication_30seed_result

PAPER_TESTS := \
	tests.test_paper_ablation_pipeline \
	tests.test_adult_taxon_ablation_pipeline \
	tests.test_condition_variant_cache \
	tests.test_cache_maintenance \
	tests.test_data_transforms \
	tests.test_config_validation \
	tests.test_models \
	tests.test_training_losses \
	tests.test_holdout_visual_notebook \
	tests.test_publication_30seed_pipeline

.PHONY: help ablation-pipeline paper-report holdout-visual-report adult-taxon-ablation-pipeline adult-taxon-report publication-pipeline publication-resolution-gapfill publication-resolution-gapfill-submit publication-data-metrics publication-resume publication-status publication-report test

help: ## Show the paper-pipeline commands.
	@echo "Worm Species paper pipeline"
	@echo
	@echo "  make ablation-pipeline                    Render the complete pipeline."
	@echo "  make ablation-pipeline PIPELINE_MODE=submit"
	@echo "                                             Submit its dependency chain."
	@echo "  make paper-report                         Rebuild completed-run paper outputs."
	@echo "  make holdout-visual-report                Build main and supplementary figures."
	@echo "  make adult-taxon-ablation-pipeline        Dry-run Adult/Juvenile combinations."
	@echo "  make adult-taxon-ablation-pipeline ADULT_TAXON_PIPELINE_MODE=submit"
	@echo "                                             Submit its 360-fit dependency chain."
	@echo "  make adult-taxon-report                   Rebuild taxon-stage ablation figures."
	@echo "  make publication-pipeline                 Dry-run the confirmed 1,890-fit pipeline."
	@echo "  make publication-resolution-gapfill       Dry-run only five new resolutions (150 fits)."
	@echo "  make publication-resolution-gapfill-submit"
	@echo "                                             Explicitly submit only those 150 fits."
	@echo "  make publication-data-metrics              Recover full-test target metrics without training."
	@echo "  make publication-pipeline PUBLICATION_PIPELINE_MODE=submit"
	@echo "                                             Submit it explicitly."
	@echo "  make publication-resume                   Explicitly resubmit; completed run IDs skip."
	@echo "  make publication-status                   Read local completion state only."
	@echo "  make publication-report                   Build all publication figures and metadata."
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

holdout-visual-report: ## Build the notebook's main and supplementary figures.
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=.:src $(PYTHON) \
		scripts/build_holdout_visual_notebook.py \
		--paper-result "$(PAPER_RESULT)" \
		--taxon-stage-result "$(ADULT_TAXON_RESULT)" \
		--output-dir "$(HOLDOUT_VISUAL_RESULT)" \
		--visual-model "$(HOLDOUT_VISUAL_MODEL)" \
		--species-ablation "$(SPECIES_ABLATION)" \
		--split-root "$(SPLIT_ROOT)" --data-root "$(DATA_ROOT)"

adult-taxon-ablation-pipeline: ## Render or submit Adult/Juvenile combination ablations.
	PYTHONPATH=.:src $(PYTHON) scripts/run_ablation_pipeline.py \
		--pipeline "$(ADULT_TAXON_PIPELINE_CONFIG)" \
		--mode "$(ADULT_TAXON_PIPELINE_MODE)"

adult-taxon-report: ## Rebuild Adult/Juvenile combination figures and source tables.
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=.:src $(PYTHON) \
		scripts/build_adult_taxon_ablation_results.py \
		--paper-result "$(ADULT_TAXON_RESULT)" \
		--split-root "$(SPLIT_ROOT)" \
		--data-root "$(DATA_ROOT)" \
		--style "$(REPORT_STYLE)"

publication-pipeline: ## Dry-run by default; submit only with an explicit mode.
	PYTHONPATH=.:src $(PYTHON) scripts/run_ablation_pipeline.py \
		--pipeline "$(PUBLICATION_PIPELINE_CONFIG)" \
		--mode "$(PUBLICATION_PIPELINE_MODE)"

publication-resolution-gapfill: ## Validate/dry-run only the five missing resolution levels.
	PYTHONPATH=.:src $(PYTHON) scripts/run_missing_resolution_losses.py

publication-resolution-gapfill-submit: ## Explicitly submit only the five missing resolution levels.
	PYTHONPATH=.:src $(PYTHON) scripts/run_missing_resolution_losses.py --mode submit

publication-data-metrics: ## Recover precision/recall/F1 from completed full-test predictions.
	PYTHONPATH=.:src $(PYTHON) scripts/augment_data_ablation_metrics.py \
		--result-root "$(PUBLICATION_RESULT)"

publication-resume: ## Resubmit the pipeline; completed best-checkpoint runs skip safely.
	PYTHONPATH=.:src $(PYTHON) scripts/run_ablation_pipeline.py \
		--pipeline "$(PUBLICATION_PIPELINE_CONFIG)" --mode submit

publication-status: ## Read completed/failed run state without querying Slurm.
	@for stage in baseline visual_ablation visual_interactions adult_taxon_baseline adult_taxon_holdouts; do \
		PYTHONPATH=.:src $(PYTHON) -m worm_species.slurm status \
			--results-root "$(PUBLICATION_RESULT)/runs/$$stage" --no-scheduler; \
	done

publication-report: ## Build all test-only figures and publication records.
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=.:src $(PYTHON) \
		scripts/build_publication_bundle.py \
		--paper-result "$(PUBLICATION_RESULT)" \
		--split-root "$(SPLIT_ROOT)" --data-root "$(DATA_ROOT)" \
		--style "$(REPORT_STYLE)"

test: ## Run the retained paper-pipeline verification surface.
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=.:src $(PYTHON) -m unittest $(PAPER_TESTS)
