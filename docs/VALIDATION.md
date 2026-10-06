# Local validation

Checked on 6 October 2026 using the existing `wormspecies` environment. This branch is for saved-result analysis and optional local inference; it does not provide neural-network training.

- All 16 numerical/protocol tests passed. These check scoring, sampling, calibration algebra, whole-worm splits, nested subsets, input/output separation, local artifact discovery and the local command interface.
- Python compilation and inference-module imports passed.
- All 4,320 task-specific visual scores were rebuilt from local saved decisions; frozen scoring definitions remain in `configs/visual_conditions.yaml` without training setup.
- `make all` dry-run contains analysis, adaptive-rank calibration, saved-control compilation, supplementary tables, plots and verification. It contains no neural-network training or image inference.
- The analysis/calibration implementations are retained from `paper-oublish`. The parent package checked baseline summary values, task-specific visual scores, five-fold allocation, saved calibration decisions and paired biological-bootstrap contrasts against the reference outputs.
- Training commands, scheduler modules, submission templates and cluster configuration were removed. The saved-control compiler reads the configured local result folder and preserves configuration/receipt validation.

The additional saved-result rerun could not proceed because the source archive at its configured mount was unavailable. No new scientific results are claimed. Full image inference, segmentation, a fresh dependency installation and a complete rerun of all 30-seed calibrations were not performed during this simplification. Images, checkpoints and saved result archives are separate inputs. Diagnostic plots are not a promise to reproduce the manuscript's arranged layout.
