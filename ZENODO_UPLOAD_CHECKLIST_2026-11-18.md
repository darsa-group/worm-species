# Release checklist

The upload files are in `/home/devd/worm-species/zenodo_release_2026-11-18`. The source code is in the repository root and is linked to `https://github.com/darsa-group/worm-species`. Online publication remains with the user.

## Prepared locally

- [x] One `images.zip` containing 9,356 original and 9,295 segmented photographs. Its size is 59,655,628,894 bytes. All 18,651 member hashes and CRCs match the original verified photographs.
- [x] One global `metadata.csv`, README with field descriptions, CC BY 4.0 licence and checksums. The image record has five upload files totalling 59,664,184,165 bytes. The user has allocated 70 GB.
- [x] Exclude calibration photographs, masks, per-image metadata and other generated processing outputs. Preserve failed segmentations, whole-worm identities and the original reference splits.
- [x] Four model upload files containing exactly the segmentation checkpoint and RGB ConvNeXt-Base seed-2340 checkpoint, settings, labels and licences.
- [x] Integrate runnable training, inference, analysis and figure code into this repository. Simplify the README and Make commands, and add comments explaining the processing and biological grouping.
- [x] Keep new test/CI files out of the public code package. Preserve the existing research scripts and tests, original README, earlier Makefile, datasets and results.
- [x] Provide an explicit commit file list and source manifest. Keep upload ZIPs, model files, raw data and generated results out of that list.
- [x] Preserve the two year ZIPs under the local preparation folder as a fallback. Upload one image-packaging variant only.

## Before publication

- [ ] Enter creators, affiliations and ORCIDs on Zenodo.
- [x] Add the supplied dataset DOI `10.5281/zenodo.24341462` and model DOI `10.5281/zenodo.24341741` to the configuration, README and release notes.
- [ ] Rebuild the final source archive, review the selected diff, commit and push. Follow `docs/RELEASE_FREEZE.md`.
- [ ] Run the freeze check and tag the checked commit as `v1.0.0`. Keep the tag fixed after publication.
- [x] The user reports that `01-images/` has been uploaded to Zenodo. The local files are now kept unchanged. Upload completion and publication were not independently checked.
- [ ] Upload the four files from `02-models/` to the model record. Use `UPLOAD_FILES.csv` to check filenames, byte sizes and hashes.
- [ ] Publish the GitHub release and both Zenodo records together under the Earthworm image classification community, then verify all links.

No files were staged or committed and no push, pull, fetch, tag, upload or publication was performed during preparation.
