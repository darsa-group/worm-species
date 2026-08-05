# 30-seed publication pipeline task list

- [x] Fix the design to seeds 40, 140, ..., 2940; weights genus=1, species=0.5, age=2; hierarchy loss h=0.
- [x] Configure Figure 1 for ConvNeXt-Base, ViT-B/16, and ResNet-50.
- [x] Configure Figures 2-6 for ConvNeXt-Base only.
- [x] Apply every visual condition during training, validation, and test.
- [x] Add binary foreground-mask-only training and testing.
- [x] Select the best checkpoint by total weighted validation loss and retain no last checkpoint.
- [x] Limit W&B to train/validation loss, validation task macro-F1, learning rate, and final test summaries.
- [x] Add test-only mean row-normalized ConvNeXt-Base confusion matrices to Figure 1.
- [x] Add Figure 7 with five reproducibly sampled test worms and all representative transforms.
- [x] Save exact best-checkpoint test predictions and confusion matrices.
- [x] Build a publication bundle with figures, figure sources, checkpoint/config/split/label-map checksums, environment, and Git provenance.
- [x] Make reruns skip completed run IDs with a successful status and retained best checkpoint.
- [ ] Run the 1,740 fits on Genome (not performed by repository validation).
- [ ] Rebuild the final bundle after every stage is complete and visually inspect all seven figures.
