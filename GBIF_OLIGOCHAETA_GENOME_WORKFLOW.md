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
`/faststorage/project/worm-species/source`. The approved jobs use the existing
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
make gbif-oligochaeta-transfer GBIF_TRANSFER_WORKERS=4
```

The dry-run is offline. The real target performs the fast structural gate and
then uses four resumable `rsync` workers without `--delete` (adjust
`GBIF_TRANSFER_WORKERS` if the login node has a different policy). It does not
recompute image hashes, so transfer progress starts after the manifest/file-
existence check and SSH connection. `transfer/FILES.txt` restricts the copy to
active iNaturalist images plus download and manifest provenance; other
publisher images are not transferred. Rerunning the command resumes an
interrupted copy. Build the training cache directly on Genome after transfer;
do not copy the cache from the local machine. A later size-and-modification-
time comparison, also without content hashing, is:

```bash
make gbif-oligochaeta-transfer-verify
```

After pulling this branch and activating `wormspecies` on Genome, inspect the
complete dependency graph without submitting:

```bash
cd /faststorage/project/worm-species/source
make gbif-train-dry-run
```

The dry-run replaces the former five-command preparation sequence. It selects
the three publication checkpoints used for inference, freezes the prepared
manifests, renders the 36-task inference array, renders the CPU cache job, and
renders both training waves without calling `sbatch`.

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

The unified command scans the completed 30-seed publication baseline directory
for `convnext_base`, `vit_b_16`, and `resnet50`. For inference only, it selects
the completed original-condition checkpoint with the lowest saved validation
loss for each backbone. It records every candidate and the deterministic
selection before submitting one combined 36-task array (three models × twelve
hash shards), capped at twelve active one-GPU tasks. A dependent CPU merge
validates exact coverage and checkpoint identity separately for each model.
Agreement with GBIF occurrence labels is not independently verified image-level
accuracy.

The old publication checkpoints are not used to initialise transfer training.
All new transfer stages start from the matching torchvision ImageNet backbone.
The four fixed-budget trajectories are:

1. `gbif_only`: ImageNet → 10,000 GBIF steps;
2. `peti_to_gbif`: ImageNet → 10,000 Petri steps → 10,000 GBIF steps;
3. `gbif_to_peti`: ImageNet → 10,000 GBIF steps → 10,000 Petri steps; and
4. `mixed`: ImageNet → 20,000 balanced steps. Its batch is always split
   exactly in half by domain; the default batch of 256 gives 128 GBIF and 128
   Petri images per step.

Every stage uses a fresh AdamW optimiser and a stage-local warmup/cosine
scheduler. Sequential Stage 2 loads only the validation-selected Stage-1 model
weights, never historical optimiser or scheduler state. Every stage finishes
its full budget and retains both the final-step `last_model.pt` and the highest
validation-score `best_model.pt`. The best checkpoint is evaluated on both
fixed test domains. This makes the saved Stage-1 and Stage-2 tests a direct
retention/catastrophic-forgetting comparison.

Checkpoint selection is trajectory-specific and excludes missing task metrics:

- GBIF selection score = mean of genus and species macro-F1;
- Petri selection score = mean of genus, species, and age macro-F1;
- `gbif_only`, `peti_to_gbif` Stage 2, and `gbif_to_peti` Stage 1 use GBIF
  validation only;
- `peti_to_gbif` Stage 1 and `gbif_to_peti` Stage 2 use Petri validation only;
- `mixed` uses the equal-weight mean of the GBIF and Petri domain scores.

GBIF age is missing and remains `NA`; it never enters a mean as zero. Each
trajectory is run with hierarchy-consistency weights 0.0 and 0.5, seeds 40,
140, and 240, for all three backbones: 72 final models and 108 total stage jobs.
Training and inference have independent, configurable Slurm resources. The
defaults are one GPU, 16 CPUs, 20 GB memory, and four hours for training versus
one GPU, 12 CPUs, 16 GB memory, and two hours for inference. Both use
`gpu-short`, `gpu-l40s`, or `gpu-h200`, with twelve active tasks maximum. The
training batch is configurable and defaults to 256; it must be even so mixed
training can derive an exact 50/50 domain split.

The 16-CPU cache sbatch job and inference array start without a dependency and
can run concurrently. Wave 1 waits for both the cache and inference merge;
Wave 2 waits for all Wave-1 tasks. One command submits the complete graph:

```bash
make gbif-train
```

All generated artifacts are contained below
`/faststorage/project/worm-species/source/outputs/gbif_training_3backbone_fixed_budget`,
including the persistent cache, manifests, Slurm scripts and logs, inference,
checkpoints, metrics, notebook exports, figures, and source tables. The curated
GBIF image bundle remains a read-only input at its configured data path.

The cache job converts every prepared GBIF
and Petri image once to a lossless 224-pixel PNG cache on shared storage. Each
physical compute node then copies that verified cache under `/tmp` once using a
node-wide `flock`; later array tasks on the same node reuse it. Training fails
closed if the cache identity, ready marker, manifest coverage, or copied image
count is invalid. Checkpoints and metrics still write to shared storage. The
submission command creates the preprocessing dependency automatically; to
build or inspect the persistent cache separately, use `make gbif-cache` and
`make gbif-status`.

`make gbif-status` reports selection, per-backbone inference, cache readiness,
and every training stage. `make gbif-resume GBIF_PHASE=primary` remains the
explicit skip-safe recovery command after failed jobs. Resume refuses to run
while a previous receipt still has active Slurm jobs, submits only incomplete
array indices, and checks completion again inside each array task before
staging the shared cache to node-local storage.

The optional legacy DINOv3 phase remains separately renderable; it is not part
of the 72-trajectory PETI↔GBIF experiment. Do not submit it unless explicitly
needed.

```bash
make gbif-dino-dry-run
make gbif-dino
make gbif-resume GBIF_PHASE=dino
```

W&B logs scalar training, validation, test, domain, model, regime, stage, and
seed metadata. Checkpoints remain on Genome and are never uploaded as W&B model
artifacts. Set `WANDB_MODE=offline` when compute nodes cannot contact W&B.

After inference and the transfer runs finish, build and execute the single
editable results notebook:

```bash
make gbif-report
```

It writes and executes `notebooks/gbif_inference_training_dino_results.ipynb`.
The notebook is the authoritative results artifact: every figure is displayed
there with its methods, checkpoint-selection rule, seed count, uncertainty,
chance provenance where applicable, and exported source table. It reports the
three old-checkpoint inference baselines separately from new transfer training,
paired hierarchy-loss effects, and Stage-1→Stage-2 retention/forgetting.
Missing jobs are displayed as pending and are never represented as results.

For each completed checkpoint, run the common evaluator (image-level PETI and
GBIF metrics plus GBIF occurrence-level aggregation) with:

```bash
make gbif-evaluate GBIF_CHECKPOINT=/absolute/path/to/best_model.pt \
  GBIF_EVALUATION_OUTPUT=/absolute/path/to/evaluation
```

The dataset-audit notebook remains unexecuted. Generate or explicitly execute it
with `make gbif-oligochaeta-notebook` and
`make gbif-oligochaeta-notebook-execute`, respectively.
