.DEFAULT_GOAL := help

PYTHON ?= python
CONFIG ?= configs/experiments/standard.yaml
CLUSTER ?= configs/clusters/local.yaml
TRAIN_CONFIG ?= config.yaml
RESULTS_ROOT ?= outputs_slurm
MAX_ACTIVE ?=
MODEL ?=
PROFILE ?=
ARTIFACTS_DIR ?= slurm/generated/plan-$(shell date -u +%Y%m%dT%H%M%S%N)
DASHBOARD_INDEX ?= .cache/worm-species-dashboard/index.sqlite3

SLURM = PYTHONPATH=src $(PYTHON) -m worm_species.slurm
CLUSTER_ARG = $(if $(strip $(CLUSTER)),--cluster-config "$(CLUSTER)",)
PLAN_OVERRIDE_VALUES = slurm.paths.results_root=$(RESULTS_ROOT) \
	$(if $(strip $(MAX_ACTIVE)),slurm.array.max_active=$(MAX_ACTIVE),) \
	$(if $(strip $(MODEL)),model.name=$(MODEL),) \
	$(if $(strip $(PROFILE)),slurm.planning.training_profile=$(PROFILE),)
PLAN_OVERRIDE_ARGS = --override "slurm.paths.results_root=$(RESULTS_ROOT)" \
	$(if $(strip $(MAX_ACTIVE)),--override "slurm.array.max_active=$(MAX_ACTIVE)",) \
	$(if $(strip $(MODEL)),--override "model.name=$(MODEL)",) \
	$(if $(strip $(PROFILE)),--override "slurm.planning.training_profile=$(PROFILE)",)
TRAIN_PROFILE_ARG = $(if $(strip $(PROFILE)),--profile "$(PROFILE)",)
TRAIN_OVERRIDE_VALUES = sweep.enabled=false \
	colour_ablation.enabled=false \
	matched_condition_training.enabled=false \
	$(if $(strip $(MODEL)),model.name=$(MODEL),) \
	$(if $(filter-out file undefined,$(origin RESULTS_ROOT)),output.out_dir=$(RESULTS_ROOT),)
TRAIN_OVERRIDE_ARGS = $(if $(strip $(TRAIN_OVERRIDE_VALUES)),--override $(TRAIN_OVERRIDE_VALUES),)

.PHONY: help validate inspect dry-run train submit status collect dashboard \
	test test-unit test-contracts test-integration clean-generated

help: ## Show the supported repository commands.
	@echo "Worm Species commands"
	@echo
	@echo "  make validate           Validate the experiment and cluster plan."
	@echo "  make inspect            Print resolved configuration and run counts."
	@echo "  make dry-run            Render a plan without scheduler submission."
	@echo "  make train              Run one canonical local training command."
	@echo "  make submit             Explicitly render and submit to SLURM."
	@echo "  make status             Summarise filesystem and scheduler status."
	@echo "  make collect            Re-run canonical result aggregation."
	@echo "  make dashboard          Launch the read-only local dashboard."
	@echo "  make test               Run the complete CPU-only unittest suite."
	@echo "  make test-unit          Run focused unit tests."
	@echo "  make test-contracts     Run compatibility and behaviour contracts."
	@echo "  make test-integration   Run lightweight integration tests."
	@echo "  make clean-generated    Remove only generated plans/local indexes."
	@echo
	@echo "Variables: CONFIG CLUSTER TRAIN_CONFIG RESULTS_ROOT MAX_ACTIVE MODEL"
	@echo "           PROFILE ARTIFACTS_DIR DASHBOARD_INDEX PYTHON"

validate: ## Validate configuration and prove the plan is internally consistent.
	@PYTHONPATH=src $(PYTHON) -c 'import sys; from worm_species.slurm.config import load_submission_config; from worm_species.slurm.planning import plan_submission; config = load_submission_config(sys.argv[1], sys.argv[2] or None, sys.argv[3:]); plan = plan_submission(config); print(f"valid: {plan.experiment_type}; {plan.array_size} task(s); profile={plan.training_profile}; cluster={plan.cluster_profile}")' \
		"$(CONFIG)" "$(CLUSTER)" $(PLAN_OVERRIDE_VALUES)

inspect: ## Print the resolved submission configuration and concise plan summary.
	@PYTHONPATH=src $(PYTHON) -c 'import sys, yaml; from worm_species.slurm.config import load_submission_config; from worm_species.slurm.planning import plan_submission; config = load_submission_config(sys.argv[1], sys.argv[2] or None, sys.argv[3:]); plan = plan_submission(config); print(yaml.safe_dump({"plan": {"experiment_type": plan.experiment_type, "cluster_profile": plan.cluster_profile, "training_profile": plan.training_profile, "models": list(plan.models), "condition_count": len(plan.conditions), "total_run_count": plan.array_size, "internal_runs_per_task": plan.expected_internal_training_runs_per_task}, "resolved_config": config}, sort_keys=False).rstrip())' \
		"$(CONFIG)" "$(CLUSTER)" $(PLAN_OVERRIDE_VALUES)

dry-run: ## Render self-contained SLURM artifacts without calling sbatch.
	$(SLURM) launch --dry-run --config "$(CONFIG)" $(CLUSTER_ARG) \
		$(PLAN_OVERRIDE_ARGS) --artifacts-dir "$(ARTIFACTS_DIR)"

train: ## Run one local process through the canonical trainer.
	PYTHONPATH=src $(PYTHON) train.py --config "$(TRAIN_CONFIG)" \
		$(TRAIN_PROFILE_ARG) --single-run $(TRAIN_OVERRIDE_ARGS)

submit: ## Explicitly render and submit the validated plan to SLURM.
	$(SLURM) launch --submit --config "$(CONFIG)" $(CLUSTER_ARG) \
		$(PLAN_OVERRIDE_ARGS) --artifacts-dir "$(ARTIFACTS_DIR)"

status: ## Summarise jobs and filesystem-derived run state.
	PYTHONPATH=src $(PYTHON) -m worm_species.slurm status \
		--results-root "$(RESULTS_ROOT)"

collect: ## Aggregate existing results without retraining.
	PYTHONPATH=src $(PYTHON) -m worm_species.slurm collect \
		--results-root "$(RESULTS_ROOT)"

dashboard: ## Launch the read-only Streamlit result browser.
	PYTHONPATH=src:. streamlit run dashboard/app.py -- \
		--results-root "$(RESULTS_ROOT)" --cache "$(DASHBOARD_INDEX)"

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

clean-generated: ## Delete only repository-local generated plans and indexes.
	@root="$$(realpath -m slurm/generated)"; \
		expected="$$(realpath -m "$(CURDIR)/slurm/generated")"; \
		test "$$root" = "$$expected"; \
		if test -d "$$root"; then find "$$root" -mindepth 1 -maxdepth 1 ! -name .gitignore -exec rm -rf -- {} +; fi
	@cache="$$(realpath -m "$(DASHBOARD_INDEX)")"; \
		cache_root="$$(realpath -m "$(CURDIR)/.cache")"; \
		case "$$cache" in "$$cache_root"/*) rm -f -- "$$cache" "$$cache-shm" "$$cache-wal" ;; \
		*) echo "Refusing to remove dashboard index outside $(CURDIR)/.cache: $$cache" >&2; exit 2 ;; esac
