# Legacy SLURM compatibility audit

The eight historical root launchers remain public symlinks to their complete
implementations under `scripts/slurm/`. None can safely be replaced by a thin
canonical wrapper yet without changing externally visible scheduling or storage
behaviour. The canonical planner is the preferred path for new submissions, but
preserving old filenames currently requires preserving their implementations.

| Public launcher | Canonical scientific mapping | Current disposition | Main parity blocker |
| --- | --- | --- | --- |
| `01_build_persistent_cache_resolved.sh` | persistent-cache maintenance | retain full implementation | no canonical cache-build command or exact lock/manifest/rebuild contract |
| `02_submit_sweep_cache_to_tmp_resolved.sh` | standard + Genome cache + `masked_hloss` | retain full implementation | historical resource, cache-copy, path, and environment defaults |
| `run_persistent_cache_sweep_wandb.sh` | standard + Genome cache + `masked_hloss_wandb` | retain full implementation | the preceding contracts plus exact W&B propagation |
| `submit_colour_ablation_sweep.sh` | colour ablation + GHPC node-local | retain full implementation | unique scratch/results, profiling, environment surface, and exact collector |
| `submit_dual_cue_experiment.sh` | dual cue + GHPC node-local | retain full implementation | custom generator/collector hooks, unique paths, profiling, and exit contracts |
| `submit_dual_cue_experiment_genome.sh` | dual cue + Genome job-local cache | retain full implementation | arbitrary base/scripts, dynamic paths, W&B/CUDA/export, and receipt layout |
| `submit_worm_node_local_scratch_sweep.sh` | standard + GHPC node-local + `masked` | retain full implementation | explicit nodes, unique scratch initialization, profiling, paths, and exits |
| `submit_worm_node_local_scratch_sweep_hloss.sh` | standard + GHPC node-local + `masked_hloss` | retain full implementation | same operational contracts as the masked launcher |

## Contracts that prevent immediate wrapper conversion

- Historical launchers normally use exit code 1 for preflight failures; the
  canonical CLI uses 2 for configuration/render errors and 4 for submission
  errors.
- Historical result roots are timestamped and contain launcher settings, run
  specifications, generated scripts, logs, submitted-job tables, and run
  copybacks. Canonical plan artifacts are deliberately separate.
- GHPC launchers require explicit GPU nodes, initialize unique node scratch,
  submit setup jobs per node, and run per-node cleanup. Reusing a stable READY
  cache would change stale-scratch behaviour.
- Cache marker names, copy filters, profiling files, CUDA diagnostics, Conda
  preflights, custom script hooks, W&B fields, and free-form SBATCH arguments are
  externally observable.
- `BASE_CONFIG` can name an arbitrary historical base file; the canonical
  experiment overlay currently extends the repository base configuration.

Scientific expansion is not the blocker: both systems preserve one run spec per
array task, disable internal re-expansion, and retain the 224-run dual-cue plan.

## Safe future migration order

1. Add a declarative legacy environment/path resolver with snapshot tests.
2. Add an exact persistent-cache maintenance command and temporary-tree tests.
3. Migrate the Genome dual-cue launcher first, then persistent-cache sweeps.
4. Migrate GHPC standard, hierarchy, dual-cue, and colour launchers only after
   unique scratch, profiling, collectors, path layout, and exit codes match.

Until those contracts exist, changing the historical launchers would violate the
behaviour-preservation requirement. They therefore remain intentional
compatibility implementations rather than uncompleted cosmetic cleanup.
