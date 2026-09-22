.DEFAULT_GOAL := help

PYTHON ?= /home/devd/miniconda3/envs/wormspecies/bin/python
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
	@echo "  make dataset-setup                       Install the local segmentation environment."
	@echo "  make dataset-plan                        Inspect image counts; write nothing."
	@echo "  make dataset DEVICE=cuda:0                Copy both years and segment both cameras."
	@echo "  make dataset-prepare                     Only combine images and write per-folder CSVs."
	@echo "  make dataset-segment DEVICE=cuda:0        Resume segmentation in the combined dataset."
	@echo "  make dataset-gpu-check                   Check PyTorch can use the NVIDIA GPU."
	@echo "  make dataset-notebook                    Rebuild the dataset analysis notebook."
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
	@echo "  make publication-test-plan                Validate 90 old checkpoints and both new cameras."
	@echo "  make publication-test-submit              Submit external testing on Genome (no training)."
	@echo "  make publication-test-status              Check file-based external-test completion."
	@echo "  make publication-test-report              Rebuild the completed external-test report."
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

# Combined publication dataset. Inputs are always read-only; the output is new.
OLD_DATASET ?= /mnt/extssd/Earthworms/petridish-worm-images
NEW_DATASET ?= /mnt/extssd/images
DATASET_ROOT ?= /mnt/extssd/Earthworms/publication_dataset
SEGMENT_MODEL ?= segment/best.pt
SEGMENT_ENV ?= .venv-segmentation
SEGMENT_PYTHON ?= $(SEGMENT_ENV)/bin/python
SEGMENT_LIB_DIR ?= $(shell "$(PYTHON)" -c 'import sys; print(sys.prefix + "/lib")')
DEVICE ?= cuda:0
SEGMENT_THREADS ?= 6
SEGMENT_LIMIT ?= 0
DATASET_ARGS = --old-root "$(OLD_DATASET)" --new-root "$(NEW_DATASET)" \
	--split-root "$(CURDIR)/split_csv" --output "$(DATASET_ROOT)"
SEGMENT_RUN = LD_LIBRARY_PATH="$(SEGMENT_LIB_DIR):$${LD_LIBRARY_PATH:-}" \
	YOLO_CONFIG_DIR="$(CURDIR)/.cache/segmentation" \
	MPLCONFIGDIR="$(CURDIR)/.cache/segmentation/matplotlib" \
	"$(SEGMENT_PYTHON)" -u
SEGMENT_ARGS = --model "$(SEGMENT_MODEL)" --device "$(DEVICE)" \
	--threads "$(SEGMENT_THREADS)" --limit "$(SEGMENT_LIMIT)"

.PHONY: dataset-setup dataset-plan dataset dataset-prepare dataset-segment dataset-gpu-check test-dataset

dataset-setup: ## Create an isolated environment using the existing PyTorch installation.
	"$(PYTHON)" -m venv --system-site-packages "$(SEGMENT_ENV)"
	"$(SEGMENT_PYTHON)" -m pip install 'ultralytics==8.4.158'
	@mkdir -p .cache/segmentation/matplotlib

dataset-plan: ## Inspect the actual input manifests without copying or segmenting.
	"$(PYTHON)" scripts/build_publication_dataset.py --mode plan $(DATASET_ARGS)

dataset: ## Prepare the combined dataset and segment both new camera folders; resumable.
	$(SEGMENT_RUN) scripts/build_publication_dataset.py --mode all $(DATASET_ARGS) $(SEGMENT_ARGS)

dataset-prepare: ## Copy source images, reuse historical segmentations, write per-folder CSVs.
	"$(PYTHON)" -u scripts/build_publication_dataset.py --mode prepare $(DATASET_ARGS)

dataset-segment: ## Segment pending images and refresh CSVs, without recopying the sources.
	$(SEGMENT_RUN) scripts/build_publication_dataset.py --mode segment $(DATASET_ARGS) $(SEGMENT_ARGS)

dataset-gpu-check: ## Fail clearly if the driver or CUDA-enabled PyTorch is unavailable.
	$(SEGMENT_RUN) -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable: fix the NVIDIA driver or set DEVICE=cpu"; print(torch.cuda.get_device_name(0))'

test-dataset: ## Verify dataset metadata, preservation, and segmentation geometry.
	$(SEGMENT_RUN) -m unittest discover -s tests -p test_publication_dataset.py

.PHONY: dataset-notebook
dataset-notebook: ## Generate the standalone notebook; reads data only when its cells run.
	"$(PYTHON)" scripts/build_publication_dataset_notebook.py

# Run from the Genome checkout with its activated wormspecies environment.
EXTERNAL_PYTHON ?= python
EXTERNAL_DATASET ?= /faststorage/project/worm-species/publication_dataset
EXTERNAL_RESULT ?= publication_external_2026
EXTERNAL_YEAR ?= 2026
EXTERNAL_BATCH_SIZE ?= 64
EXTERNAL_WORKERS ?= 6
EXTERNAL_CPUS ?= 8
EXTERNAL_MEMORY ?= 32G
EXTERNAL_TIME ?= 02:00:00
EXTERNAL_MAX_ACTIVE ?= 12
EXTERNAL_ACCOUNT ?= worm-species
EXTERNAL_PARTITION ?= gpu-short
EXTERNAL_REPORT_CPUS ?= 4
EXTERNAL_REPORT_MEMORY ?= 8G
EXTERNAL_REPORT_TIME ?= 00:30:00
EXTERNAL_REPORT_PARTITION ?=
EXTERNAL_ARGS = --dataset "$(EXTERNAL_DATASET)" --publication-results "$(PUBLICATION_RESULT)" \
	--output "$(EXTERNAL_RESULT)" --year "$(EXTERNAL_YEAR)" \
	--batch-size "$(EXTERNAL_BATCH_SIZE)" --workers "$(EXTERNAL_WORKERS)" \
	--cpus "$(EXTERNAL_CPUS)" --memory "$(EXTERNAL_MEMORY)" --time "$(EXTERNAL_TIME)" \
	--max-active "$(EXTERNAL_MAX_ACTIVE)" --account "$(EXTERNAL_ACCOUNT)" \
	--partition "$(EXTERNAL_PARTITION)" --report-cpus "$(EXTERNAL_REPORT_CPUS)" \
	--report-memory "$(EXTERNAL_REPORT_MEMORY)" --report-time "$(EXTERNAL_REPORT_TIME)" \
	--report-partition "$(EXTERNAL_REPORT_PARTITION)"

.PHONY: publication-test-plan publication-test-submit publication-test-status publication-test-report test-publication-external
publication-test-plan: ## Freeze input metadata and render Slurm commands without submitting.
	"$(EXTERNAL_PYTHON)" scripts/run_publication_external_test.py plan $(EXTERNAL_ARGS)

publication-test-submit: ## Submit pending checkpoint evaluations and a dependent CPU report.
	"$(EXTERNAL_PYTHON)" scripts/run_publication_external_test.py submit $(EXTERNAL_ARGS)

publication-test-status: ## Inspect completed artifacts, not live Slurm state.
	"$(EXTERNAL_PYTHON)" scripts/run_publication_external_test.py status --output "$(EXTERNAL_RESULT)"

publication-test-report: ## Collect all 30 seeds per model after both cameras finish.
	"$(EXTERNAL_PYTHON)" scripts/run_publication_external_test.py collect --output "$(EXTERNAL_RESULT)"

test-publication-external: ## Test external evaluation, checkpoint validation, resume, and reporting.
	MPLCONFIGDIR=/tmp/mplconfig "$(EXTERNAL_PYTHON)" -m unittest discover -s tests -p test_publication_external_test.py
