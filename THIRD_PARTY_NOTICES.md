# Third-party components

Original project code is supplied under MIT. Dependencies and model components retain their own terms. This repository does not relicense them.

The optional segmentation commands use Ultralytics. Its upstream software and trained segmentation model use AGPL-3.0 by default. The source of the project wrapper and modified image processing is included in `scripts/release.py` and `segment/inference_coco.py`. The full upstream licence is in `licenses/ULTRALYTICS-AGPL-3.0.txt`. Obtain the matching Ultralytics source for the installed version from [the upstream repository](https://github.com/ultralytics/ultralytics). Its [licensing page](https://www.ultralytics.com/license) describes its terms for trained models. Distribution or use of a derivative segmentation application must comply with the applicable upstream terms; the MIT notice for original code does not remove them.

The classifier uses torchvision models and preprocessing. Retain `licenses/TORCHVISION-BSD.txt`. Other Python dependencies remain subject to their respective distributed notices.

The two checkpoint files are distributed in the companion Zenodo model record. The project rights in the ConvNeXt checkpoint and model documentation use CC BY 4.0. The Ultralytics segmentation component retains its upstream terms. The companion image record uses CC BY 4.0 for photographs and metadata. See each record's licence guide for the exact scope.
