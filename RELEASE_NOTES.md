# Earthworm image classification 1.0.1

Code: https://github.com/darsa-group/worm-species/releases/tag/v1.0.1

Image dataset: [10.5281/zenodo.24341462](https://doi.org/10.5281/zenodo.24341462).

Model weights: [10.5281/zenodo.24341741](https://doi.org/10.5281/zenodo.24341741).

This release provides the code for image segmentation, classifier training, inference, specimen-level testing, statistical analyses and publication figures. The image dataset contains 9,356 original photographs, 9,295 segmented photographs and one global metadata table. The model record provides the segmentation checkpoint and an RGB ConvNeXt-Base classifier trained with seed 2340. Masks and other processing outputs can be generated through code. Reproducing the full study comparisons requires retraining the omitted classifier fits.

Retraining the segmentation model requires the original human annotations and training split, which are outside the image release.

Original code uses MIT. Retain the third-party licence notices. Data and model licences are provided in their Zenodo records.
