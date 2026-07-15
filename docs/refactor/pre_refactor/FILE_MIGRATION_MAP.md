# File migration map

No legacy path was removed. Root entry points and historical `src.*` modules
remain compatibility surfaces.

| Old path or embedded responsibility | Canonical implementation | Old-path status |
| --- | --- | --- |
| `src.utils.load_config` | `src/worm_species/config/loading.py` | re-exported from `src.utils` |
| `src.utils.parse_scalar`, `set_nested`, `apply_overrides` | `src/worm_species/config/overrides.py` | re-exported from `src.utils` |
| sweep parsing/product duplicated in ordinary trainers | `src/worm_species/config/sweeps.py` | imported names remain public in each trainer |
| colour-ablation sweep expansion duplicated in colour/cue trainers | `src/worm_species/config/sweeps.py` | legacy trainer functions wrap the canonical expansion with colour mode enabled |
| `src.models` implementation | `src/worm_species/models/factory.py` | `src/models.py` is a compatibility re-export |
| `MultiTaskClassifier` and `build_multitask_model` duplicated in five trainers | `src/worm_species/models/multitask.py` | imported names remain public from every trainer |
| label-map construction duplicated in five trainers | `src/worm_species/data/labels.py` | imported `build_label_maps` remains public from every trainer |
| predefined split CSV reader duplicated in five trainers | `src/worm_species/data/labels.py` | imported `read_csvs_from_dir` remains public; path semantics unchanged |
| `generate_dual_cue_run_specs.py` condition logic | `src/worm_species/experiments/conditions.py` | root script is a thin wrapper and re-exports former functions |
| `generate_dual_cue_run_specs.py` writing/CLI | `src/worm_species/experiments/run_specs.py` | root positional CLI preserved |
| `collect_dual_cue_results.py` | `src/worm_species/experiments/result_collection.py` | root positional CLI preserved and former functions re-exported |

## Intentionally not moved

- `src/dataset.py` and `src/dataset_multitask.py`: similar names hide different
  single-task/multi-task and scientific behavior.
- ordinary, hierarchy, colour-ablation, and cue-suppression epoch/training
  loops: hierarchy and experiment variants require more differential fixtures.
- matched-condition training and fixed-RGB stress evaluation: deliberately
  remain distinct paths inside the cue workflow.
- root shell/SLURM launchers: preserved byte-for-byte during this refactor.
- notebooks: no path, cell, import, output, or execution assumption changed.
- any generated, output, cache, checkpoint, W&B, data, or scratch tree.
