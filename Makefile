.DEFAULT_GOAL := help

PYTHON ?= python
EXPERIMENT ?= standard
CONFIG ?= configs/experiments/$(EXPERIMENT).yaml
CLUSTER ?= configs/clusters/genome.yaml
TRAIN_CONFIG ?= config.yaml
RESULTS_ROOT ?= output_allrun
SLURM_RESULTS_ROOT ?= $(RESULTS_ROOT)
SINGLE_TASK_RESULTS_ROOT ?= single_task/outputs
MAX_ACTIVE ?=
MODEL ?=
ARTIFACTS_DIR ?= logs/generated/plan-$(shell date -u +%Y%m%dT%H%M%S%N)
DASHBOARD_INDEX ?= .cache/worm-species-dashboard/index.sqlite3
DASHBOARD_DERIVED ?= .cache/worm-species-dashboard/derived
WIZARD_OUTPUT ?= configs/experiments/interactive.yaml

SLURM = PYTHONPATH=src $(PYTHON) -m worm_species.slurm
CLUSTER_ARG = $(if $(strip $(CLUSTER)),--cluster-config "$(CLUSTER)",)
PLAN_OVERRIDE_VALUES = slurm.paths.results_root=$(RESULTS_ROOT) \
	$(if $(strip $(MAX_ACTIVE)),slurm.array.max_active=$(MAX_ACTIVE),) \
	$(if $(strip $(MODEL)),model.name=$(MODEL),)
PLAN_OVERRIDE_ARGS = --override "slurm.paths.results_root=$(RESULTS_ROOT)" \
	$(if $(strip $(MAX_ACTIVE)),--override "slurm.array.max_active=$(MAX_ACTIVE)",) \
	$(if $(strip $(MODEL)),--override "model.name=$(MODEL)",)
TRAIN_OVERRIDE_VALUES = sweep.enabled=false \
	colour_ablation.enabled=false \
	matched_condition_training.enabled=false \
	$(if $(strip $(MODEL)),model.name=$(MODEL),) \
	$(if $(filter-out file undefined,$(origin RESULTS_ROOT)),output.out_dir=$(RESULTS_ROOT),)
TRAIN_OVERRIDE_ARGS = $(if $(strip $(TRAIN_OVERRIDE_VALUES)),--override $(TRAIN_OVERRIDE_VALUES),)

.PHONY: help configure validate inspect dry-run train submit status collect dashboard-prepare dashboard \
	test test-unit test-contracts test-integration clean-generated run-dev

help: ## Show the supported repository commands.
	@echo "Worm Species commands"
	@echo
	@echo "  make configure          Open the interactive experiment builder."
	@echo "  make validate           Validate the experiment and cluster plan."
	@echo "  make inspect            Print resolved configuration and run counts."
	@echo "  make dry-run            Render a plan without scheduler submission."
	@echo "  make train              Run one canonical local training command."
	@echo "  make submit             Explicitly render and submit to SLURM."
	@echo "  make status             Summarise filesystem and scheduler status."
	@echo "  make collect            Re-run canonical result aggregation."
	@echo "  make dashboard-prepare  Prepare cached metrics and confusion matrices."
	@echo "  make dashboard          Launch the read-only local dashboard."
	@echo "  make test               Run the complete CPU-only unittest suite."
	@echo "  make test-unit          Run focused unit tests."
	@echo "  make test-contracts     Run compatibility and behaviour contracts."
	@echo "  make test-integration   Run lightweight integration tests."
	@echo "  make clean-generated    Remove only generated plans/local indexes."
	@echo "  make run-dev Clean the output folder and run all configs in dev folder"
	@echo
	@echo "Variables: CONFIG CLUSTER TRAIN_CONFIG RESULTS_ROOT MAX_ACTIVE MODEL"
	@echo "           EXPERIMENT (standard, hierarchy, dual_cue, colour_ablation,"
	@echo "                       patch_shuffle_matrix, persistent_hierarchy)"
	@echo "           SLURM_RESULTS_ROOT SINGLE_TASK_RESULTS_ROOT ARTIFACTS_DIR"
	@echo "           DASHBOARD_INDEX DASHBOARD_DERIVED WIZARD_OUTPUT PYTHON"
	@echo
	@echo "Examples:"
	@echo "  make dry-run EXPERIMENT=patch_shuffle_matrix"
	@echo "  make submit EXPERIMENT=dual_cue CLUSTER=configs/clusters/genome.yaml"
	@echo "  make train MODEL=convnext_base"

configure: ## Open the arrow-key terminal experiment builder.
	PYTHONPATH=src $(PYTHON) -m worm_species.config.tui --output "$(WIZARD_OUTPUT)"

validate: ## Validate configuration and prove the plan is internally consistent.
	@$(SLURM) validate --config "$(CONFIG)" $(CLUSTER_ARG) $(PLAN_OVERRIDE_ARGS)

inspect: ## Print the resolved submission configuration and concise plan summary.
	@$(SLURM) inspect --config "$(CONFIG)" $(CLUSTER_ARG) $(PLAN_OVERRIDE_ARGS)

dry-run: ## Render self-contained SLURM artifacts without calling sbatch.
	$(SLURM) launch --dry-run --config "$(CONFIG)" $(CLUSTER_ARG) \
		$(PLAN_OVERRIDE_ARGS) --artifacts-dir "$(ARTIFACTS_DIR)"

train: ## Run one local process through the canonical trainer.
	PYTHONPATH=src $(PYTHON) -m worm_species.training --config "$(TRAIN_CONFIG)" \
		--single-run $(TRAIN_OVERRIDE_ARGS)

submit: ## Explicitly render and submit the validated plan to SLURM.
	$(SLURM) launch --submit --config "$(CONFIG)" $(CLUSTER_ARG) \
		$(PLAN_OVERRIDE_ARGS) --artifacts-dir "$(ARTIFACTS_DIR)"

status: ## Summarise jobs and filesystem-derived run state.
	PYTHONPATH=src $(PYTHON) -m worm_species.slurm status \
		--results-root "$(RESULTS_ROOT)"

collect: ## Aggregate existing results without retraining.
	PYTHONPATH=src $(PYTHON) -m worm_species.slurm collect \
		--results-root "$(RESULTS_ROOT)"

dashboard-prepare: ## Prepare cached metrics and combined confusion matrices read-only.
	PYTHONPATH=src $(PYTHON) -m worm_species.results.derive \
		--source "slurm=$(SLURM_RESULTS_ROOT)" \
		--source "single_task=$(SINGLE_TASK_RESULTS_ROOT)" \
		--cache "$(DASHBOARD_DERIVED)" --render all

dashboard: ## Launch the read-only Streamlit result browser.
	PYTHONPATH=src:. streamlit run dashboard/app.py -- \
		--source "slurm=$(SLURM_RESULTS_ROOT)" \
		--source "single_task=$(SINGLE_TASK_RESULTS_ROOT)" \
		--cache "$(DASHBOARD_INDEX)" --derived-cache "$(DASHBOARD_DERIVED)"

run-dev:
	@rm -rf $(RESULTS_ROOT)
	@rm -rf logs/generated/plan*
	@for config in dev/*.yaml; do \
		echo "Running $$config"; \
		$(MAKE) submit CONFIG="$$config"; \
	done

test: ## Run every standard-library test without external data or GPUs.
	PYTHONPATH=.:src $(PYTHON) -m unittest discover -s tests -p 'test_*.py'

test-unit: ## Run focused configuration and planning units.
	PYTHONPATH=.:src $(PYTHON) -m unittest discover -s tests -p 'test_config_validation.py'
	PYTHONPATH=.:src $(PYTHON) -m unittest discover -s tests -p 'test_slurm_planning.py'

test-contracts: ## Run public CLI, import, and scientific behaviour contracts.
	PYTHONPATH=.:src $(PYTHON) -m unittest discover -s tests -p 'test_behaviour_contracts.py'
	PYTHONPATH=.:src $(PYTHON) -m unittest discover -s tests -p 'test_training_cli_contracts.py'

test-integration: ## Run lightweight result-discovery integration tests.
	PYTHONPATH=.:src $(PYTHON) -m unittest discover -s tests -p 'test_result_discovery.py'
	PYTHONPATH=.:src $(PYTHON) -m unittest discover -s tests -p 'test_result_derivation.py'
	PYTHONPATH=.:src $(PYTHON) -m unittest discover -s tests -p 'test_dashboard_multi_root.py'

clean-generated: ## Delete only repository-local generated plans and indexes.
	@root="$$(realpath -m slurm/generated)"; \
		expected="$$(realpath -m "$(CURDIR)/slurm/generated")"; \
		test "$$root" = "$$expected"; \
		if test -d "$$root"; then find "$$root" -mindepth 1 -maxdepth 1 ! -name .gitignore -exec rm -rf -- {} +; fi
	@cache="$$(realpath -m "$(DASHBOARD_INDEX)")"; \
		cache_root="$$(realpath -m "$(CURDIR)/.cache")"; \
		case "$$cache" in "$$cache_root"/*) rm -f -- "$$cache" "$$cache-shm" "$$cache-wal" ;; \
		*) echo "Refusing to remove dashboard index outside $(CURDIR)/.cache: $$cache" >&2; exit 2 ;; esac
	@derived="$$(realpath -m "$(DASHBOARD_DERIVED)")"; \
		cache_root="$$(realpath -m "$(CURDIR)/.cache")"; \
		case "$$derived" in "$$cache_root"/*) if test -d "$$derived"; then rm -rf -- "$$derived"; fi ;; \
		*) echo "Refusing to remove derived dashboard cache outside $(CURDIR)/.cache: $$derived" >&2; exit 2 ;; esac
