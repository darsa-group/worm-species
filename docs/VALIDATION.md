# Validation of the reproduction branch

Checked locally on 6 October 2026 in the existing `wormspecies` environment. Generated verification outputs were stored separately under `/tmp`; the original manuscript, training runs and result directories were not rewritten.

## Executed

- Configuration validation and 13 numerical/protocol tests: passed. Tests cover fixed scoring with out-of-cohort predictions and abstentions, whole-worm bootstrap behaviour, correction/head algebra, view aggregation, matched training subsets, five-fold independence, nested calibration subsets, input/output separation, visual intervention order, exclusion guards and refusal of local matched-control training.
- Python compilation for all included modules: passed.
- Make dry run: `make all` contains saved-artifact analysis, calibration, CSV compilation, supplementary tables, diagnostic plotting and verification; no neural training or inference.
- CPU baseline rescoring: all **54** external-performance summary rows matched the authoritative saved analysis exactly, including point estimates and sampling intervals.
- Original visual rescoring: all **4,320** task-specific condition/seed F1 values matched the frozen analysis exactly, without using manuscript/poster files.
- Five-fold reconstruction: all worm allocations matched the authoritative calibration plan. The primary global calibration/test sets matched as well.
- Fixed-rank mapping check: held-out decisions for **15** task/method combinations (five feature methods × three tasks), seed 40, global fold 0, matched saved decisions exactly. No neural checkpoint was loaded.
- CSV-only matched-control compiler: validated all **750** completed fits using prediction CSVs, class maps, configuration hashes and completion receipts, without loading checkpoints.
- Paired whole-worm bootstrap: all **36** biological contrasts, their sampling intervals and seed SDs matched the manuscript's frozen table within numerical tolerance.
- Historical training pipeline dry run: rendered the five complete experiment stages and required image/condition-cache jobs. The recovery-only resolution subset was not duplicated. No `sbatch` submission occurred. Rendered shell syntax was checked.

## Not executed as part of packaging

No neural-network training, CUDA inference, physical-camera acquisition, live Slurm submission, new segmentation, complete rerun of every calibration fit, or fresh dependency installation was performed. These require the external archives/checkpoints and appropriate runtime. The scientific calibration core was checked against saved held-out outcomes; it was not re-estimated for all 30 seeds during packaging.

The new figure runner is a diagnostic CSV-based presentation, not a pixel-identical export of the manuscript's composed layouts. PCA/UMAP figures are qualitative and do not establish quantitative recovery. Their parameter and coordinate files are exported separately.

The code/config package does not replace the required data/checkpoint release, missing acquisition records, or manual image/mask review. Dataset/code reuse licences and archive access remain separate release decisions.
