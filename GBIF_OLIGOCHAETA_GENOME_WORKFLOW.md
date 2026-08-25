# Local GBIF download to Genome DINOv3 workflow

This workflow keeps network-heavy image acquisition on the local machine and
runs GPU embedding plus UMAP/HDBSCAN on Genome. Nothing is submitted by a
default Make target: transfer and Slurm submission each require an explicit
command.

## Why any login is needed

- A **GBIF account** is used only to request the citable asynchronous DWCA.
  It lets GBIF create a reproducible download key/DOI and send status
  notifications. Building the manifest and downloading publisher image URLs
  do not use the saved GBIF password. The credentials are environment
  variables and are never written to the manifests.
- An **SSH login to Genome** is needed only for a real `rsync`, checksum check,
  or `sbatch`. The local transfer check and both dry-runs open no connection.
- The first DINOv3 weight load may require whatever model-registry access is
  required by the installed `timm` checkpoint if it is not already cached on
  Genome. No registry token is put in this repository or the data bundle.

GBIF `StillImage` means only that a multimedia row is an image. It does not say
that the image pixels contain an earthworm rather than text, a label, a map, or
habitat. The manifest retains optional `media_title` and `media_description`,
but DINOv3 clustering and human review remain the content-quality gate.

## 1. Complete the local bundle

The allow-list is only `Crassiclitellata` and `Moniligastrida`. `Enchytraeida`
(white worms), aquatic oligochaetes, branchiobdellids, and leeches are excluded.
The first pass uses only `iNaturalist research-grade observations` (dataset key
`50c9509d-22c7-4a22-a47d-8c48425ef4a7`).
The genus-defined filter means both `genus` and `genusKey` must be non-empty in
the DWCA; it is not an image-content filter.

```bash
make gbif-oligochaeta-download-images
make gbif-oligochaeta-prune-missing-images-dry-run
# Use only when the remaining failed publisher images are intentionally excluded:
make gbif-oligochaeta-prune-missing-images
make gbif-oligochaeta-filter-dataset-dry-run
make gbif-oligochaeta-filter-dataset
make gbif-oligochaeta-transfer-check
```

The downloader is resumable. It fetches a repeated publisher URL once and maps
the verified asset back to every distinct occurrence-image relationship. The
transfer check fails unless every original media row is accounted for by either
the active downloaded manifest or `excluded_missing_images.csv`. Every active
row must be `downloaded`, have an existing file, and have a SHA-256. Excluded
rows retain their publisher error and an explicit exclusion reason; the full
GBIF media manifest is never rewritten.

## 2. Prepare the Genome checkout and environment

The code is versioned in Git, while the image bundle is deliberately ignored by
Git. Push this feature branch, then update the actual Genome runtime checkout at
`/home/devd/worm-species/source`. The approved jobs use the existing
`wormspecies` environment. Update it with the GBIF/DINO/W&B/reporting
dependencies before the first run:

```bash
conda env update --name wormspecies -f configs/gbif_oligochaeta_environment.yaml
conda activate wormspecies
```

Do not use `--prune` unless you have separately checked every dependency needed
by the wider repository. Before submitting a large job, confirm that `timm` can load the configured
DINOv3 checkpoint in that environment; the Slurm job also checks imports and
CUDA before processing images.

## 3. Transfer and verify

```bash
make gbif-oligochaeta-transfer-dry-run
make gbif-oligochaeta-transfer
```

The dry-run is offline. The real target performs the fast structural gate and
then uses resumable `rsync` without `--delete`. It does not recompute image
hashes, so transfer progress starts after the manifest/file-existence check and
SSH connection. `transfer/FILES.txt` restricts the copy to active iNaturalist
images plus download and manifest provenance; other publisher images are not
transferred. Rerunning the command resumes an interrupted copy. A later
size-and-modification-time comparison, also without content hashing, is:

```bash
make gbif-oligochaeta-transfer-verify
```

## 4. Embed and cluster on Genome

```bash
make gbif-oligochaeta-genome-dry-run
make gbif-oligochaeta-genome-submit
```

The dry-run prints the exact SSH and `sbatch` command without connecting. The
explicit submit target runs one `gpu-short` job in the `worm-species` account:

1. validate the transferred manifest and environment;
2. calculate L2-normalized DINOv3 ViT-B/16 embeddings;
3. reduce the embeddings with PCA and deterministic UMAP;
4. cluster the reduced features with HDBSCAN; and
5. save embeddings and cluster artifacts inside the transferred bundle.

The two-hour resource request follows the repository's Genome profile. It is a
runtime estimate, not a guarantee; if the real job reaches the partition limit,
retain its logs and embeddings and adjust the resource plan before resubmitting.

## 5. Review, existing model, and fine-tuning

Pull `embeddings/` and `curation/clusters.csv` back to the local bundle, then
run the review interface:

```bash
make gbif-oligochaeta-pull-genome-results
make gbif-oligochaeta-curate
```

The Streamlit interface shows the UMAP,
cluster filters, thumbnails, publisher links, and reversible content labels.
Nothing is deleted.

After exporting the review, copy the curated manifest to Genome with
`make gbif-oligochaeta-push-curation`. All later commands are run directly in
the Genome checkout; they do not SSH from another machine.

The approved experiment is configured in `configs/gbif_training.yaml`. It
keeps the existing Petri train/validation/test splits and creates a deterministic
GBIF 65/15/20 split whose groups connect both occurrence identity and exact
image duplicates. Labels supported by fewer than three independent groups are
masked for that task without discarding rows useful to another task. Exact
duplicate groups with contradictory GBIF species labels have the species task
masked and are recorded in the preparation audit. The two Petri-only exact
labels `Aporrectodea_tuberculata` and
`Lumbricus_terrestris_herculeus` remain distinct.

Run and inspect baseline inference first:

```bash
make gbif-check
make gbif-infer-dry-run GBIF_CHECKPOINT=/absolute/path/to/best_model.pt
make gbif-infer GBIF_CHECKPOINT=/absolute/path/to/best_model.pt
```

Inference is one 12-element Slurm array (`0-11%12`), not a twelve-GPU job.
Each task requests one GPU and processes a deterministic hash shard. A dependent
CPU job refuses to merge missing, duplicated, unexpected, or mixed-checkpoint
outputs. Reported agreement with GBIF occurrence labels is not independently
verified image-level accuracy.

The primary training phase runs `vit_b_16`, `resnet50`, and `convnext_base`
with five seeds. Each model uses the curated→Petri, Petri→curated, and balanced
mixed regimes. Fixed optimizer steps rather than equal epochs give every regime
the same number of image draws from each differently sized domain. Hierarchy
loss is disabled. Early stopping has a 6,000-step floor and twelve-validation
patience. Jobs request one GPU, 16 CPUs, 20 GB host memory, 12 DataLoader
workers, and may run on `gpu-short`, `gpu-l40s`, or `gpu-h200`. Array concurrency
is capped at 12.

```bash
make gbif-train-dry-run
make gbif-train
make gbif-status
make gbif-resume GBIF_PHASE=primary
```

Only after the primary phase, run the three-seed DINOv3 ViT-B/16 phase:

```bash
make gbif-dino-dry-run
make gbif-dino
make gbif-resume GBIF_PHASE=dino
```

W&B logs scalar training, validation, test, domain, model, regime, stage, and
seed metadata. Checkpoints remain on Genome and are never uploaded as W&B model
artifacts. Set `WANDB_MODE=offline` when compute nodes cannot contact W&B.

After inference and both training phases finish, build and execute the single
editable results notebook:

```bash
make gbif-report
```

It writes `notebooks/gbif_inference_training_dino_results.ipynb` and exports
source tables plus SVG/PDF figures below the configured output root. Missing
jobs are displayed as pending and are never represented as scientific results.

The dataset-audit notebook remains unexecuted. Generate or explicitly execute it
with `make gbif-oligochaeta-notebook` and
`make gbif-oligochaeta-notebook-execute`, respectively.
