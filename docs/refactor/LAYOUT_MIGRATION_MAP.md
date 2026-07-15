# Layout migration map

This pass changes repository layout only. Scientific code paths, output names,
checkpoints, splits, and live result trees are outside its scope.

## Entry points

| Historical path | Canonical path | Compatibility |
| --- | --- | --- |
| `train_multitask_*.py` | `scripts/training/` | root Python wrappers re-export the canonical modules and call the same `main()` |
| `generate_dual_cue_run_specs.py` | `scripts/training/generate_dual_cue_run_specs.py` | root Python wrapper |
| `collect_dual_cue_results.py` | `scripts/maintenance/collect_dual_cue_results.py` | root Python wrapper |
| cache, sweep, colour, cue, and node-local shell launchers | `scripts/slurm/` | root symbolic links preserve the exact shell content |
| `archive_project_to_drive.sh` | `scripts/maintenance/` | root symbolic link |
| `tranfser_from_genome.sh`, `tranfser_from_ghpc.sh` | `scripts/transfer/` | misspelled root paths retained as symbolic links |

## Supporting files

| Historical location | Canonical location | Compatibility |
| --- | --- | --- |
| `config_old.yaml` | `configs/config_old.yaml` | root symbolic link |
| `environment.yaml` | `configs/environment.yaml` | root symbolic link |
| root and legacy-directory notebooks | `notebooks/{analysis,diagnostics,interpretability,data}/` | relative repository paths anchored to the discovered root where required |
| `refactor_audit*` | `docs/refactor/{compatibility,pre_refactor,post_refactor}/` | documentation references updated; no runtime reader depended on old paths |
| `refactor_contract_*` | `docs/refactor/contracts/{before,after}/` | comparison commands use the new paths |
| generated cue atlas PNGs | `figures/cue_suppression/` | generating notebook writes to the new directory |

The four root `worm_build_cache_*.out/.err` symbolic links remain at their old
paths. Their targets are relative links to an external SSD, and future names
are controlled by existing SLURM directives; moving them would not be a safe
layout-only operation.

The externally linked `split_csv/*.csv` files and every `outputs_slurm` path
remain untouched.
