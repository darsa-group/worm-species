# Behaviour contract results

## Automated contracts

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Result: 15 tests passed.

Covered contracts:

- scalar parsing, deep-copy dotted overrides, stable run hashing/naming;
- ordinary Cartesian sweep order;
- saturation sequence and equivalent endpoint deduplication;
- one- and two-model dual-cue run specs;
- generated matched-condition expansion disabling and launcher internal-sweep
  disabling;
- matched train/validation/test transform ordering;
- deterministic fixed-RGB transformed-test transforms;
- missing-label recognition and training-only sorted class maps;
- predefined split path and CSV read semantics;
- hierarchy child-to-parent mapping and consistency loss;
- evaluation metric key set and macro-F1;
- multi-task head output shapes and state-dict parameter names;
- ordinary, colour, and cue checkpoint top-level schemas;
- matched-versus-fixed-RGB result join columns/calculation;
- five legacy training CLI help flag surfaces;
- all root shell script syntax;
- notebook JSON/static code-cell compatibility.

## Baseline-to-refactor CLI comparison

The seven public Python entry points were extracted from the baseline commit
with `git archive`. Their baseline and refactored `--help` output was compared
byte-for-byte with the repository root on `PYTHONPATH`.

- `collect_dual_cue_results.py`: unchanged;
- `generate_dual_cue_run_specs.py`: unchanged;
- all five `train_multitask_*.py` entry points: unchanged.

## Baseline-to-refactor run-spec comparison

The dual-cue generator was extracted from the actual baseline commit with
`git archive`. Baseline and refactored generators each produced 224 run specs
from `config.yaml`.

- recursive byte comparison: no differences;
- supplied `compare_contract_trees.py`: no file-tree/schema differences.

The retained trees are `../contracts/before/` and
`../contracts/after/`.

## Static validation after each extraction

- `python -m compileall -q` passed for changed Python modules and entry points.
- a no-bytecode static compile passed for all 38 bounded Python source files.
- `git diff --check` passed.
- root shell scripts passed `bash -n` from the contract suite.

The environment emits a non-fatal `libtinfo.so.6` version warning when spawning
`bash` from the active Conda environment.

## Not executed

- full training on the image dataset;
- GPU/AMP numerical equivalence;
- SLURM submission, array scheduling, scratch setup/copy-back, or cleanup;
- W&B online/offline run creation or artifact upload;
- clean-kernel notebook execution;
- full fixed-checkpoint cue battery on a real checkpoint.

No claim of GPU, HPC, or full scientific numerical equivalence is made.

## Concurrent split-file change

Final `git status` reported mode changes for `split_csv/train_split.csv`,
`val_split.csv`, and `test_split.csv`: regular tracked CSVs became symbolic
links to an external SSD at 13:03 local time. No refactor command wrote these
paths. They were left untouched because restoring or accepting changed split
membership requires an explicit scientific decision.
