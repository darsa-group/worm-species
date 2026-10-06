# Local validation

Checked on 6 October 2026 using the existing `wormspecies` environment.

- All 20 tests passed: numerical/protocol checks plus frozen training counts, matching visual condition definitions, protected output paths, partial-fit handling and an actual one-epoch synthetic CPU fit.
- The CPU fit used an unpretrained, frozen-backbone ResNet-18. Its saved checkpoint was reloaded strictly, predictions were obtained, and its saved probabilities passed the same biological-control validator used by the analysis. The control used reference-cohort individual class weights even after training worms were removed. This is an interface test, not a scientific experiment.
- A full plan against the mounted prepared dataset produced 2,640 configurations: 90 architecture baselines, 840 visual conditions, 600 visual interactions, 30 biological references, 330 biological exclusions and 750 matched controls. No paper training fit was launched. Original partitions remain worm-disjoint. Control and downstream-analysis test-split hashes agree.
- Representative existing best checkpoints were strictly loaded on CPU for all three baseline architectures and for visual, combined-visual, biological-exclusion and matched-control runs. Synthetic forward passes produced finite logits with the saved head sizes. This tests compatibility, not classification performance.
- All 4,320 task-specific visual scores were previously rebuilt from local saved decisions. The frozen scoring definitions remain consistent with the restored training conditions.
- `make all` remains saved-result analysis only; training is exposed separately through `train-plan`, `train`, `train-all` and `train-status`. No cluster configuration, submission templates or remote-login commands are included.

Full publication-model training, full image inference/segmentation, an installation in a fresh environment, online W&B connectivity and a complete rerun of every 30-seed analysis were not performed in this update. Existing research results are preserved. Dataset, checkpoints and saved results are separate inputs; manuscript figures are supplied separately from the diagnostic plotting commands.
