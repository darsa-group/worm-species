# GBIF Oligochaeta image curation and transfer plan

## Goal

Build a provenance-preserving image corpus for true earthworms only, use
pretrained DINOv3 embeddings and interactive cluster review to
remove unsuitable images, measure how the existing worm classifier transfers
to overlapping taxa, and only then fine-tune and run a selected model over the
curated corpus.

## Taxonomic scope

GBIF's current backbone resolves the requested class name `Oligochaeta` as a
synonym of accepted class `Clitellata` (class key `255`). Querying the synonym
key `8166676` does not retrieve the accepted concept's descendants, while
querying all of `Clitellata` also includes leeches. The acquisition query must
therefore use an explicit, versioned two-order allow-list rather than silently
using either class key. The user-approved scope is `Crassiclitellata` plus
`Moniligastrida`. The first analysis pass is restricted to GBIF dataset
`iNaturalist research-grade observations`
(`50c9509d-22c7-4a22-a47d-8c48425ef4a7`). Enchytraeida (white worms), aquatic oligochaetes,
branchiobdellids, and leeches are excluded.

Every included and excluded order name and key is written to the acquisition
audit before a download is requested.

## Stages and gates

1. **Audit the scope.** Resolve every configured taxon key against GBIF and
   capture the response and query count. Fail if a key resolves to an
   unexpected name or rank.
2. **Request a citable occurrence download.** Submit an authenticated GBIF
   `DWCA` request for still images, the allowed order keys, and a non-null genus
   key. Store the request, download key, DOI, and response metadata. Credentials
   are read from environment variables and never written to disk.
3. **Build the media manifest.** Join the occurrence core to the multimedia
   extension, re-check that both genus and genus key are present, and emit one
   row per distinct image URL, including every still image when one occurrence
   has multiple images. Preserve occurrence, dataset, taxon, creator, reference,
   and licence fields. Audit expected versus downloaded media counts per
   occurrence and fail the download command while any image remains failed.
   Never treat a GBIF taxon label as image-level truth
   without retaining its identification metadata.
4. **Download images resumably.** Stream publisher-hosted images, cap file
   size, verify with Pillow, hash content, identify exact duplicates, and retain
   failures in the manifest. If permanently unavailable publisher images are
   intentionally excluded, move their rows into a separate exclusion audit and
   reconcile active plus excluded IDs against the unchanged source manifest.
   Files are never removed by the curation workflow.
5. **Embed and cluster.** Use one pinned DINOv3/timm checkpoint and its resolved
   inference transform. Save L2-normalized embeddings, a row index, model
   provenance, deterministic UMAP coordinates, and HDBSCAN cluster labels.
6. **Curate interactively.** Review clusters and individual images in
   Streamlit. Store decisions separately from source metadata using controlled
   labels such as `keep`, `reject_text`, `reject_non_organism`,
   `reject_duplicate`, and `unsure`. Exporting the curated manifest is
   reversible and does not delete source images.
7. **Run the existing classifier.** Select a real original-image best
   checkpoint and use its embedded config and label maps. Report exact species
   overlap, genus-only overlap, and out-of-vocabulary taxa before inference.
   Never score unknown taxa as though the checkpoint was trained for them.
8. **Fine-tune.** Create grouped train/validation partitions that keep images
   from the same occurrence and duplicate group together. Start with a frozen
   backbone and new heads, then compare controlled partial/full fine-tuning.
   Select on validation only and retain a fixed final evaluation partition.
9. **Run the selected model over the curated corpus.** Export probabilities,
   predictions, known/unknown scope, checkpoint identity, and source manifest
   hashes. Do not overwrite baseline predictions.

## Completion boundaries

Repository tests and small synthetic DWCA fixtures validate code paths only.
They do not establish a completed GBIF download, valid publisher media URLs,
GPU embedding throughput, model accuracy, or fine-tuning results. Large
downloads and GPU/cluster jobs are explicit commands and are not defaults.
