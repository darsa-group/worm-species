# Notebooks

These notebooks are active analysis and diagnostic code. Run Jupyter with the
repository root as its working directory so saved run paths and presentation
artefacts resolve consistently:

```bash
PYTHONPATH=src python -m jupyter lab --notebook-dir .
```

Notebooks that import project code locate `config.yaml` and
`src/worm_species/`, add `src/` to `sys.path`, and import the canonical
`worm_species` package. They fail clearly when the repository root cannot be
located. `src.splits` and `src.cache` remain temporary bridges where no
canonical package equivalent exists. Historical single-task CAM and UMAP cells
retain `src.dataset.prepare_metadata` because that implementation is part of
their saved-checkpoint data contract and is not interchangeable with the
multi-task metadata rules.

## Inputs and outputs

Experiment directories under `outputs/` and `outputs_slurm/`, saved
`config.json` files, checkpoints, split CSVs, datasets, and W&B artefacts are
read-only notebook inputs. Running a notebook must not modify them.

New presentation artefacts are written to workflow-specific directories:

| Notebook group | Typical inputs | New artefacts |
| --- | --- | --- |
| `analysis/` | completed run metrics, histories, reports, condition summaries | `figures/<workflow>/`, `tables/<workflow>/` |
| `diagnostics/` | completed run outputs and recreated held-out predictions | `figures/<diagnostic>/`, `tables/<diagnostic>/` |
| `interpretability/` | saved configs, label maps, images, and checkpoints | `figures/<method>/`, `tables/<method>/` |
| `data/` | external metadata/images and the existing `split_csv/` links | display-only unless the notebook declares a destination |

Existing generated artefacts remain where they are. Path changes affect only
future notebook saves.

## Runtime requirements

The result-summary notebooks generally need pandas, NumPy, Matplotlib, and
IPython. Image diagnostics and interpretability notebooks may additionally need
Pillow, OpenCV, scikit-learn, PyTorch, torchvision, tqdm, Plotly, and
`umap-learn`. Checkpoint-backed notebooks require the referenced checkpoint and
dataset paths; GPU use is optional unless a notebook-specific workload is too
large for practical CPU execution.

Several notebooks intentionally contain explicit historical run or external
dataset paths near the top of the first code cell. Update those user settings
deliberately for a different experiment. They are not inferred from whichever
directory happens to contain the notebook.

## Validation

Static notebook contracts do not load datasets or checkpoints:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
  python -m unittest -q tests.test_notebook_migration
```

The contracts validate the tracked inventory, JSON structure, Python code-cell
syntax, canonical imports, repository bootstrap, output routing, and preserved
scientific path/filename literals. Full clean-kernel execution is an optional
integration check because it needs external data, checkpoints, and optional
visualisation dependencies.

`interpretability/cam_mutlitask.ipynb` is a historical zero-byte placeholder,
and `interpretability/single_task_umap.ipynb` is an intentionally empty valid
notebook. Both remain unchanged.
