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
GBIF_CHECKPOINT ?= /faststorage/project/worm-species/source/publication_30seed_result/runs/baseline/run_012_convnext_base_original_loss_genus-1.0_species-0.5_age-2.0_seed_1240_hloss_0.0_20260805153055/lr_0.0003_hloss_False_f7435e3a_train_original/best_model.pt
GBIF_CURATED_MANIFEST ?= gbif_oligochaeta/curation/curated_manifest.csv
GBIF_EXISTING_PREDICTIONS ?= gbif_oligochaeta/predictions/existing_checkpoint.csv
GBIF_DATASET_NOTEBOOK ?= notebooks/gbif_earthworm_dataset_audit.ipynb
GBIF_NOTEBOOK_TIMEOUT ?= -1
GBIF_DOWNLOAD_KEY ?=
GBIF_BUNDLE_ROOT ?= gbif_oligochaeta
GBIF_GENOME_REMOTE ?= devd@login.genome.au.dk
GBIF_GENOME_DATA_ROOT ?= /faststorage/project/worm-species/data/gbif_oligochaeta
GBIF_GENOME_PROJECT_ROOT ?= /faststorage/project/worm-species/source
GBIF_GENOME_CONDA_ENV ?= wormspecies
GBIF_TRANSFER_WORKERS ?= 4
GBIF_TRAINING_CONFIG ?= configs/gbif_training.yaml
GBIF_FULL_TAXONOMY_CONFIG ?= configs/gbif_full_taxonomy.yaml
GBIF_EVALUATION_OUTPUT ?=
GBIF_PHASE ?= primary
GBIF_PYTHON ?= python

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
	tests.test_publication_30seed_pipeline \
	tests.test_gbif_oligochaeta \
	tests.test_gbif_domain_experiment \
	tests.test_gbif_transfer_analysis \
	tests.test_gbif_full_taxonomy \
	tests.test_gbif_combined_results_notebook

.PHONY: help ablation-pipeline paper-report holdout-visual-report adult-taxon-ablation-pipeline adult-taxon-report publication-pipeline publication-resolution-gapfill publication-resolution-gapfill-submit publication-data-metrics publication-resume publication-status publication-report gbif-oligochaeta-scope gbif-oligochaeta-audit-scope gbif-oligochaeta-request gbif-oligochaeta-download-status gbif-oligochaeta-download-dwca gbif-oligochaeta-manifest gbif-oligochaeta-download-images gbif-oligochaeta-prune-missing-images-dry-run gbif-oligochaeta-prune-missing-images gbif-oligochaeta-filter-dataset-dry-run gbif-oligochaeta-filter-dataset gbif-oligochaeta-transfer-check gbif-oligochaeta-transfer-dry-run gbif-oligochaeta-transfer gbif-oligochaeta-transfer-verify gbif-oligochaeta-pull-genome-results gbif-oligochaeta-push-curation gbif-oligochaeta-genome-dry-run gbif-oligochaeta-genome-submit gbif-oligochaeta-embed gbif-oligochaeta-cluster gbif-oligochaeta-curate gbif-oligochaeta-infer-existing gbif-oligochaeta-notebook gbif-oligochaeta-notebook-execute gbif-check gbif-prepare gbif-cache gbif-infer-dry-run gbif-infer gbif-train-dry-run gbif-train gbif-dino-dry-run gbif-dino gbif-status gbif-resume gbif-evaluate gbif-report gbif-transfer-analysis-dry-run gbif-transfer-analysis gbif-full-taxonomy-dry-run gbif-full-taxonomy test

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
	@echo "  make gbif-oligochaeta-scope               Print the explicit GBIF taxon scope."
	@echo "  make gbif-oligochaeta-audit-scope         Resolve scope and current counts against GBIF."
	@echo "  make gbif-oligochaeta-request             Request the authenticated GBIF DWCA."
	@echo "  make gbif-oligochaeta-download-dwca GBIF_DOWNLOAD_KEY=..."
	@echo "                                             Download a completed DWCA."
	@echo "  make gbif-oligochaeta-manifest            Build a media manifest from the saved DWCA."
	@echo "  make gbif-oligochaeta-download-images     Resume and require every image download."
	@echo "  make gbif-oligochaeta-prune-missing-images-dry-run"
	@echo "                                             Report rows without usable files."
	@echo "  make gbif-oligochaeta-prune-missing-images"
	@echo "                                             Exclude those rows from the active manifest."
	@echo "  make gbif-oligochaeta-filter-dataset-dry-run"
	@echo "                                             Preview the iNaturalist-only filter."
	@echo "  make gbif-oligochaeta-filter-dataset       Apply the iNaturalist-only filter."
	@echo "  make gbif-oligochaeta-transfer-check      Refuse unless the local bundle is complete."
	@echo "  make gbif-oligochaeta-transfer-dry-run    Show the transfer without connecting."
	@echo "  make gbif-oligochaeta-transfer            Start resumable parallel rsync without hashing."
	@echo "  make gbif-oligochaeta-transfer-verify     Compare file size/time without hashing."
	@echo "  make gbif-oligochaeta-pull-genome-results Pull embeddings/clusters for review."
	@echo "  make gbif-oligochaeta-push-curation       Push the reviewed manifest to Genome."
	@echo "  make gbif-oligochaeta-genome-dry-run      Show the Genome Slurm command only."
	@echo "  make gbif-oligochaeta-genome-submit       Explicitly submit DINOv3 + UMAP."
	@echo "  make gbif-oligochaeta-embed               Run DINOv3 over downloaded images."
	@echo "  make gbif-oligochaeta-cluster             Cluster saved DINOv3 embeddings."
	@echo "  make gbif-oligochaeta-curate              Open the interactive curation app."
	@echo "  make gbif-oligochaeta-infer-existing GBIF_CHECKPOINT=/path/best_model.pt"
	@echo "                                             Run a real existing checkpoint."
	@echo "  make gbif-oligochaeta-notebook            Regenerate the dataset-audit notebook."
	@echo "  make gbif-oligochaeta-notebook-execute    Explicitly execute it in place."
	@echo "  make gbif-check                          Validate the approved Genome config."
	@echo "  make gbif-cache                          Explicitly build/reuse preprocessed images."
	@echo "  make gbif-infer GBIF_CHECKPOINT=/path/best_model.pt"
	@echo "                                             Submit 12 one-GPU inference shards."
	@echo "  make gbif-train                          Submit 3-model inference, cache, and 72 trajectories."
	@echo "  make gbif-evaluate GBIF_CHECKPOINT=...   Evaluate one completed checkpoint."
	@echo "  make gbif-dino                           Submit optional legacy DINO arrays."
	@echo "  make gbif-status                         Read output completion state."
	@echo "  make gbif-resume GBIF_PHASE=primary      Resubmit skip-safe incomplete work."
	@echo "  make gbif-report                         Build the combined results notebook."
	@echo "  make gbif-transfer-analysis-dry-run      Validate/render completed-run analysis."
	@echo "  make gbif-transfer-analysis              Submit inference + 128-core report job."
	@echo "  make gbif-full-taxonomy-dry-run          Render the new immutable three-phase DAG."
	@echo "  make gbif-full-taxonomy                  Submit audit → training → inference → report."
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

gbif-oligochaeta-scope: ## Print the reviewed, explicit GBIF taxon scope.
	PYTHONPATH=.:src $(PYTHON) scripts/gbif_oligochaeta_pipeline.py scope

gbif-oligochaeta-audit-scope: ## Verify configured keys and current image counts against GBIF.
	PYTHONPATH=.:src $(PYTHON) scripts/gbif_oligochaeta_pipeline.py audit-scope

gbif-oligochaeta-request: ## Request a GBIF DWCA using GBIF_USERNAME/PASSWORD/EMAIL.
	PYTHONPATH=.:src $(PYTHON) scripts/gbif_oligochaeta_pipeline.py request-download

gbif-oligochaeta-download-status: ## Inspect an asynchronous GBIF download.
	@test -n "$(GBIF_DOWNLOAD_KEY)" || (echo "Set GBIF_DOWNLOAD_KEY" >&2; exit 2)
	PYTHONPATH=.:src $(PYTHON) scripts/gbif_oligochaeta_pipeline.py download-status "$(GBIF_DOWNLOAD_KEY)"

gbif-oligochaeta-download-dwca: ## Download a completed GBIF archive.
	@test -n "$(GBIF_DOWNLOAD_KEY)" || (echo "Set GBIF_DOWNLOAD_KEY" >&2; exit 2)
	PYTHONPATH=.:src $(PYTHON) scripts/gbif_oligochaeta_pipeline.py download-dwca "$(GBIF_DOWNLOAD_KEY)"

gbif-oligochaeta-manifest: ## Join a completed GBIF DWCA into one row per image.
	PYTHONPATH=.:src $(PYTHON) scripts/gbif_oligochaeta_pipeline.py build-manifest

gbif-oligochaeta-download-images: ## Resume images and fail until every media row succeeds.
	PYTHONPATH=.:src $(PYTHON) scripts/gbif_oligochaeta_pipeline.py download-images

gbif-oligochaeta-prune-missing-images-dry-run: ## Report unusable rows without changing files.
	PYTHONPATH=.:src $(PYTHON) scripts/gbif_oligochaeta_pipeline.py prune-missing-images

gbif-oligochaeta-prune-missing-images: ## Keep only usable rows and retain an exclusion audit.
	PYTHONPATH=.:src $(PYTHON) scripts/gbif_oligochaeta_pipeline.py \
		prune-missing-images --apply

gbif-oligochaeta-filter-dataset-dry-run: ## Preview the configured GBIF dataset filter.
	PYTHONPATH=.:src $(PYTHON) scripts/gbif_oligochaeta_pipeline.py filter-dataset

gbif-oligochaeta-filter-dataset: ## Keep only the configured GBIF dataset active.
	PYTHONPATH=.:src $(PYTHON) scripts/gbif_oligochaeta_pipeline.py filter-dataset --apply

gbif-oligochaeta-transfer-check: ## Validate completeness locally; no network connection.
	PYTHONPATH=.:src $(PYTHON) scripts/prepare_gbif_earthworm_transfer.py \
		--bundle-root "$(GBIF_BUNDLE_ROOT)"

gbif-oligochaeta-transfer-dry-run: ## Validate and print transfer commands without SSH.
	PYTHON="$(PYTHON)" scripts/transfer_gbif_earthworms_to_genome.sh \
		--mode dry-run --bundle-root "$(GBIF_BUNDLE_ROOT)" \
		--remote "$(GBIF_GENOME_REMOTE)" --remote-path "$(GBIF_GENOME_DATA_ROOT)" \
		--workers "$(GBIF_TRANSFER_WORKERS)"

gbif-oligochaeta-transfer: ## Explicitly transfer the bundle without content hashing.
	PYTHON="$(PYTHON)" scripts/transfer_gbif_earthworms_to_genome.sh \
		--mode transfer --bundle-root "$(GBIF_BUNDLE_ROOT)" \
		--remote "$(GBIF_GENOME_REMOTE)" --remote-path "$(GBIF_GENOME_DATA_ROOT)" \
		--workers "$(GBIF_TRANSFER_WORKERS)"

gbif-oligochaeta-transfer-verify: ## Compare local/remote size and mtime without hashing.
	PYTHON="$(PYTHON)" scripts/transfer_gbif_earthworms_to_genome.sh \
		--mode verify --bundle-root "$(GBIF_BUNDLE_ROOT)" \
		--remote "$(GBIF_GENOME_REMOTE)" --remote-path "$(GBIF_GENOME_DATA_ROOT)"

gbif-oligochaeta-pull-genome-results: ## Pull embeddings and clusters for local review.
	PYTHON="$(PYTHON)" scripts/transfer_gbif_earthworms_to_genome.sh \
		--mode pull-results --bundle-root "$(GBIF_BUNDLE_ROOT)" \
		--remote "$(GBIF_GENOME_REMOTE)" --remote-path "$(GBIF_GENOME_DATA_ROOT)"

gbif-oligochaeta-push-curation: ## Push the reviewed manifest and decisions to Genome.
	PYTHON="$(PYTHON)" scripts/transfer_gbif_earthworms_to_genome.sh \
		--mode push-curation --bundle-root "$(GBIF_BUNDLE_ROOT)" \
		--remote "$(GBIF_GENOME_REMOTE)" --remote-path "$(GBIF_GENOME_DATA_ROOT)"

gbif-oligochaeta-genome-dry-run: ## Render the Genome submission without connecting.
	scripts/run_gbif_earthworms_on_genome.sh --mode dry-run \
		--remote "$(GBIF_GENOME_REMOTE)" \
		--project-root "$(GBIF_GENOME_PROJECT_ROOT)" \
		--bundle-root "$(GBIF_GENOME_DATA_ROOT)" \
		--conda-env "$(GBIF_GENOME_CONDA_ENV)"

gbif-oligochaeta-genome-submit: ## Explicitly submit the DINOv3 and clustering job.
	scripts/run_gbif_earthworms_on_genome.sh --mode submit \
		--remote "$(GBIF_GENOME_REMOTE)" \
		--project-root "$(GBIF_GENOME_PROJECT_ROOT)" \
		--bundle-root "$(GBIF_GENOME_DATA_ROOT)" \
		--conda-env "$(GBIF_GENOME_CONDA_ENV)"

gbif-oligochaeta-embed: ## Compute versioned DINOv3 image embeddings.
	PYTHONPATH=.:src $(PYTHON) scripts/gbif_oligochaeta_pipeline.py embed

gbif-oligochaeta-cluster: ## Reduce and HDBSCAN-cluster DINOv3 embeddings.
	PYTHONPATH=.:src $(PYTHON) scripts/gbif_oligochaeta_pipeline.py cluster

gbif-oligochaeta-curate: ## Open the reversible Streamlit cluster-review UI.
	PYTHONPATH=.:src $(PYTHON) -m streamlit run scripts/gbif_oligochaeta_curate.py

gbif-oligochaeta-infer-existing: ## Run an existing classifier on curated images.
	@test -n "$(GBIF_CHECKPOINT)" || (echo "Set GBIF_CHECKPOINT=/path/to/best_model.pt" >&2; exit 2)
	PYTHONPATH=.:src $(PYTHON) scripts/gbif_oligochaeta_pipeline.py infer-existing \
		--manifest "$(GBIF_CURATED_MANIFEST)" \
		--checkpoint "$(GBIF_CHECKPOINT)" \
		--output "$(GBIF_EXISTING_PREDICTIONS)"

gbif-oligochaeta-notebook: ## Regenerate the reproducible dataset-audit notebook.
	PYTHONPATH=.:src $(PYTHON) scripts/build_gbif_earthworm_dataset_notebook.py

gbif-oligochaeta-notebook-execute: ## Explicitly execute the existing audit notebook without regenerating it.
	@test -f "$(GBIF_DATASET_NOTEBOOK)" || (echo "Run make gbif-oligochaeta-notebook first" >&2; exit 2)
	MPLCONFIGDIR=/tmp/mplconfig JUPYTER_CONFIG_DIR=/tmp/jupyter-config \
		JUPYTER_DATA_DIR=/tmp/jupyter-data PYTHONPATH=.:src $(PYTHON) -m jupyter \
		nbconvert --to notebook --execute --inplace "$(GBIF_DATASET_NOTEBOOK)" \
		--ExecutePreprocessor.timeout=$(GBIF_NOTEBOOK_TIMEOUT)

gbif-check: ## Validate the approved direct-on-Genome experiment config.
	PYTHONPATH=.:src $(GBIF_PYTHON) scripts/gbif_domain_experiment.py \
		--config "$(GBIF_TRAINING_CONFIG)" check-config

gbif-prepare: ## Build deterministic union labels and leakage-safe GBIF splits.
	PYTHONPATH=.:src $(GBIF_PYTHON) scripts/gbif_domain_experiment.py \
		--config "$(GBIF_TRAINING_CONFIG)" prepare

gbif-cache: gbif-prepare ## Build/reuse the persistent lossless training-image cache.
	PYTHONPATH=.:src $(GBIF_PYTHON) scripts/gbif_domain_experiment.py \
		--config "$(GBIF_TRAINING_CONFIG)" build-cache

gbif-infer-dry-run: ## Render the 12-shard one-GPU inference array without sbatch.
	@test -n "$(GBIF_CHECKPOINT)" || (echo "Set GBIF_CHECKPOINT=/path/to/best_model.pt" >&2; exit 2)
	PYTHONPATH=.:src $(GBIF_PYTHON) scripts/gbif_domain_experiment.py \
		--config "$(GBIF_TRAINING_CONFIG)" render-inference \
		--checkpoint "$(GBIF_CHECKPOINT)"

gbif-infer: ## Submit inference locally from a Genome terminal; no SSH.
	@test -n "$(GBIF_CHECKPOINT)" || (echo "Set GBIF_CHECKPOINT=/path/to/best_model.pt" >&2; exit 2)
	PYTHONPATH=.:src $(GBIF_PYTHON) scripts/gbif_domain_experiment.py \
		--config "$(GBIF_TRAINING_CONFIG)" submit-inference \
		--checkpoint "$(GBIF_CHECKPOINT)"

gbif-train-dry-run: ## Discover three baselines and render the complete primary DAG.
	PYTHONPATH=.:src $(GBIF_PYTHON) scripts/gbif_domain_experiment.py \
		--config "$(GBIF_TRAINING_CONFIG)" render-primary-pipeline

gbif-train: ## Submit inference + cache + 72 final trajectories as one DAG.
	PYTHONPATH=.:src $(GBIF_PYTHON) scripts/gbif_domain_experiment.py \
		--config "$(GBIF_TRAINING_CONFIG)" submit-primary-pipeline

gbif-dino-dry-run: ## Prepare and render the later three-seed DINO arrays.
	PYTHONPATH=.:src $(GBIF_PYTHON) scripts/gbif_domain_experiment.py \
		--config "$(GBIF_TRAINING_CONFIG)" render-training --phase dino

gbif-dino: ## Submit the later three-seed DINO arrays on Genome.
	PYTHONPATH=.:src $(GBIF_PYTHON) scripts/gbif_domain_experiment.py \
		--config "$(GBIF_TRAINING_CONFIG)" submit-training --phase dino

gbif-status: ## Read direct-on-Genome result files without querying Slurm.
	PYTHONPATH=.:src $(GBIF_PYTHON) scripts/gbif_domain_experiment.py \
		--config "$(GBIF_TRAINING_CONFIG)" status

gbif-resume: ## Resubmit a skip-safe primary or DINO phase from Genome.
	@test "$(GBIF_PHASE)" = primary -o "$(GBIF_PHASE)" = dino || (echo "GBIF_PHASE must be primary or dino" >&2; exit 2)
	PYTHONPATH=.:src $(GBIF_PYTHON) scripts/gbif_domain_experiment.py \
		--config "$(GBIF_TRAINING_CONFIG)" submit-training --phase "$(GBIF_PHASE)"

gbif-evaluate: ## Evaluate one completed checkpoint on PETI and GBIF test manifests.
	@test -n "$(GBIF_CHECKPOINT)" || (echo "Set GBIF_CHECKPOINT=/path/to/best_model.pt" >&2; exit 2)
	@test -n "$(GBIF_EVALUATION_OUTPUT)" || (echo "Set GBIF_EVALUATION_OUTPUT=/path/to/evaluation" >&2; exit 2)
	PYTHONPATH=.:src $(GBIF_PYTHON) scripts/evaluate_gbif_transfer.py \
		--config "$(GBIF_TRAINING_CONFIG)" --checkpoint "$(GBIF_CHECKPOINT)" \
		--output "$(GBIF_EVALUATION_OUTPUT)" --device cuda

gbif-report: ## Generate and execute the combined inference/training results notebook.
	PYTHONPATH=.:src $(GBIF_PYTHON) scripts/build_gbif_combined_results_notebook.py \
		--config "$(GBIF_TRAINING_CONFIG)" --execute

gbif-transfer-analysis-dry-run: ## Validate and render the post-training analysis DAG.
	PYTHONPATH=.:src $(GBIF_PYTHON) scripts/analyse_gbif_transfer.py \
		--config "$(GBIF_TRAINING_CONFIG)" run --mode dry-run

gbif-transfer-analysis: ## Submit inference and its dependent 128-core analysis job.
	PYTHONPATH=.:src $(GBIF_PYTHON) scripts/analyse_gbif_transfer.py \
		--config "$(GBIF_TRAINING_CONFIG)" run --mode submit

gbif-full-taxonomy-dry-run: ## Render all phases without calling sbatch.
	PYTHONPATH=.:src $(GBIF_PYTHON) scripts/gbif_full_taxonomy_pipeline.py \
		--config "$(GBIF_FULL_TAXONOMY_CONFIG)" run --mode dry-run

gbif-full-taxonomy: ## Submit the immutable full-taxonomy audit/training/report DAG.
	PYTHONPATH=.:src $(GBIF_PYTHON) scripts/gbif_full_taxonomy_pipeline.py \
		--config "$(GBIF_FULL_TAXONOMY_CONFIG)" run --mode submit

test: ## Run the retained paper-pipeline verification surface.
	MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=.:src $(PYTHON) -m unittest $(PAPER_TESTS)
