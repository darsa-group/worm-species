# Legacy compatibility archive

This directory contains historical implementations that are no longer the
preferred development surface but must remain available for reproducibility and
existing user workflows. Nothing here should be selected by new launchers.

## Layout and compatibility paths

| Archived implementation | Preserved public path | Preferred replacement |
| --- | --- | --- |
| `slurm/01_build_persistent_cache_resolved.sh` | `scripts/slurm/01_build_persistent_cache_resolved.sh` and root alias | canonical cache maintenance is still deferred; retain this launcher when its exact cache contract is required |
| `slurm/02_submit_sweep_cache_to_tmp_resolved.sh` | matching `scripts/slurm/` and root aliases | `make submit` with an experiment and cluster profile |
| `slurm/run_persistent_cache_sweep_wandb.sh` | matching `scripts/slurm/` and root aliases | `make submit` with resolved W&B configuration |
| `slurm/submit_colour_ablation_sweep.sh` | matching `scripts/slurm/` and root aliases | `make submit CONFIG=configs/experiments/colour_ablation.yaml` |
| `slurm/submit_dual_cue_experiment.sh` | matching `scripts/slurm/` and root aliases | `make submit CONFIG=configs/experiments/dual_cue.yaml CLUSTER=configs/clusters/ghpc.yaml` |
| `slurm/submit_dual_cue_experiment_genome.sh` | matching `scripts/slurm/` and root aliases | `make submit CONFIG=configs/experiments/dual_cue.yaml CLUSTER=configs/clusters/genome.yaml` |
| `slurm/submit_worm_node_local_scratch_sweep.sh` | matching `scripts/slurm/` and root aliases | canonical standard experiment with the GHPC cluster profile |
| `slurm/submit_worm_node_local_scratch_sweep_hloss.sh` | matching `scripts/slurm/` and root aliases | canonical standard experiment with the `masked_hloss` training profile |
| `configs/config_old.yaml` | `configs/config_old.yaml` and root `config_old.yaml` | `config.yaml` plus `configs/experiments/` and `configs/clusters/` |
| `python/experiments/generate_sweep_run_specs.py` | `src/generate_sweep_run_specs.py` | `worm_species.slurm.planning` for new submission plans |

The shell and configuration compatibility paths are relative symlinks. The old
ordinary sweep generator remains a thin Python wrapper because its historical
module and command-line imports are public. These mechanisms preserve old paths
without keeping substantial deprecated bodies in active source directories.

## Intentionally not archived

- Root training, run-spec, and collection files are required thin compatibility
  wrappers.
- `scripts/training/` and the dual-cue collection wrapper expose tested public
  helper imports even though their scientific implementations are canonical.
- `src/dataset.py`, `src/cache.py`, `src/splits.py`, and `src/utils.py` still
  support notebooks, cache construction, or canonical training.
- `src/download_pretrained_from_config.py`, transfer scripts, and the archive
  utility provide unique active maintenance behavior with no replacement.
- Notebooks, split files, datasets, checkpoints, `single_task/outputs/`, and
  `outputs_slurm/` are scientific inputs, provenance, or results—not legacy
  source cleanup candidates.

The exact operational differences that prevent replacing historical SLURM
launchers with canonical wrappers are documented in
`docs/refactor/LEGACY_SLURM_COMPATIBILITY_AUDIT.md`.
