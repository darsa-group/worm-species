# Legacy archive

This directory stores historical implementations and compatibility adapters
outside the active source and command surface. New code must use the canonical
modules under `src/worm_species/`, `train.py`, the SLURM module, or the Make
targets.

Historical paths are intentionally absent. They can be recreated explicitly:

```bash
legacy/restore_compatibility.sh --dry-run
legacy/restore_compatibility.sh
```

Use `--root PATH` to restore into another checkout. Restoration preflights the
complete manifest and refuses to overwrite any different file or link. An
already restored, byte-identical tree is accepted, making the command
idempotent. There is deliberately no force option.

## Archive layout

- `compatibility/root/`: former root Python commands.
- `compatibility/scripts/`: former Python entry points below `scripts/`.
- `compatibility/src/`: historical `src.*` import and generator adapters.
- `slurm/`: exact historical SLURM launcher bodies.
- `configs/config_old.yaml`: the previous configuration body.
- `python/experiments/generate_sweep_run_specs.py`: the exact ordinary sweep
  generator retained for provenance.
- `compatibility.map`: tab-separated active path, archive source, restoration type,
  mode, hash, and canonical replacement for every restorable path.

Python adapters are restored as regular files so their `__file__` and package
behavior match the historical commands. SLURM and old-configuration aliases are
restored as their former relative symlinks.

## Preferred replacements

| Historical surface | Preferred replacement |
| --- | --- |
| `train_multitask_*.py` | `python train.py --config config.yaml` or `python -m worm_species.training` with explicit configuration switches |
| dual-cue and ordinary run-spec generators | `python -m worm_species.slurm dry-run` |
| dual-cue collector | `python -m worm_species.slurm collect` |
| historical submission scripts | `make submit` or `python -m worm_species.slurm submit` with experiment and cluster configuration |
| `config_old.yaml` | `config.yaml`, `configs/experiments/`, and `configs/clusters/` |
| `src.dataset_multitask` | responsibility-specific modules under `worm_species.data` |
| `src.models` | `worm_species.models` |

## Not legacy

Notebooks are scientific analysis code, not legacy files. The following also
remain outside this archive: split files, datasets, checkpoints,
`single_task/outputs/`, `outputs_slurm/`, figures, tables, logs, active transfer
scripts, the archive utility, `src/dataset.py`, `src/cache.py`, `src/splits.py`,
and `src/utils.py`.

The historical operational differences of the archived SLURM launchers are
documented in `docs/refactor/LEGACY_SLURM_COMPATIBILITY_AUDIT.md`.
