# Notebook compatibility

## Validation performed

- Located 19 source notebooks outside pruned generated directories.
- Parsed notebook JSON.
- Compiled Python code cells after removing IPython magic and shell-command
  lines.
- Preserved all historical `src.dataset`, `src.dataset_multitask`,
  `src.models`, `src.splits`, `src.cache`, and `src.utils` import paths.
- Did not execute notebooks that load datasets/checkpoints or write analysis
  artifacts.

## Result

Eighteen notebooks parse and statically compile. `cam_mutlitask.ipynb` is a
pre-existing zero-byte invalid JSON file and remains unchanged.

Project-importing notebooks remain compatible through old-path re-exports:

| Import surface | Representative notebooks |
| --- | --- |
| `src.dataset` | `single_task/cam.ipynb`, `single_task/umap_species_embeddings.ipynb` |
| `src.dataset_multitask` | Grad-CAM, UMAP, same-individual, and dataset-test notebooks |
| `src.models` | Grad-CAM, UMAP, and same-individual notebooks |
| `src.splits` | dataset-test and top-model Grad-CAM notebooks |
| `src.cache`, `src.utils` | `dataset_tes.ipynb` |

## Clean-kernel status

Clean-kernel execution is unverified. Several notebooks require external
metadata roots, generated sweep outputs, large checkpoints, GPUs, or optional
plotting/UMAP dependencies. Static pass is not represented as runtime
equivalence.
