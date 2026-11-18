# Commit and freeze version 1.0.0

The release workflow is in the root of `darsa-group/worm-species`. The current branch is `paper-pipeline-cache-only`. On 18 November 2026, a read-only remote-reference check found the remote branch at `0fc5c4735822a66c3460af624a6eeb60266cbda6` and remote `main` at `f730af5e999c174145bdc867c5ac6660eb0303d8`. Repeat a status check before pushing because other contributors can change GitHub meanwhile.

Earlier research scripts remain available. Use `make -f Makefile.research help` for those commands and `README.md` for the public workflow. A local copy of every overwritten file and the initial working-tree diff is under `zenodo_release_2026-11-18/preparation/before_repository_integration/`.

## Before the release commit

The dataset DOI `10.5281/zenodo.24341462` and model DOI `10.5281/zenodo.24341741` are already in `configs/release_inputs.json`, `README.md` and `RELEASE_NOTES.md`. The configured GitHub release URL is the planned `v1.0.0` URL. Creators, affiliations and ORCIDs are entered on Zenodo by the user. The prepared final archive is `zenodo_release_2026-11-18/github-source-1.0.0-final.zip`.

Use the environment installed from the README and build a fresh source archive. A fresh filename preserves earlier archive copies.

```bash
python scripts/package_source.py --output zenodo_release_2026-11-18/github-source-1.0.0-rebuilt.zip
python scripts/check_release_freeze.py --require-links
```

`configs/release_source_paths.txt` lists the files in the public source archive. `SOURCE_MANIFEST.csv` records their SHA256 hashes. `configs/release_commit_paths.txt` also includes the retained research Makefile, earlier README and this repository's commit instructions. Generated packages, images, checkpoints, results and unrelated manuscript changes are excluded from that selected commit list.

Review the currently staged changes first. If the index already contains unrelated work, handle that work before running the selected-file staging command. No files were staged or committed during preparation.

```bash
git status --short --branch
git diff --cached --stat
git add --pathspec-from-file=configs/release_commit_paths.txt
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "Prepare reproducible earthworm code release 1.0.0"
make release-freeze-check
```

The final check fails while reserved DOIs are missing, source hashes are stale or release files differ from `HEAD`. Rebuild the source archive after any edit, then commit the updated manifests with the edited files. Keep the existing image and checkpoint hashes in `configs/release_inputs.json`.

## Push and create the version tag

The following are user-run publication commands. They were not executed during preparation. Push the reviewed commit on `paper-pipeline-cache-only`, the branch retained for the paper. Run `make release-freeze-check` on that exact final release commit before tagging it.

```bash
git push origin paper-pipeline-cache-only
git tag -a v1.0.0 -m "Earthworm image classification 1.0.0"
git push origin v1.0.0
```

Create a GitHub release for `v1.0.0`, using `RELEASE_NOTES.md` and the final source ZIP. Keep the release draft until both Zenodo drafts are ready. Publish the three together, then verify their links. A GitHub repository archive is unnecessary; retain the version tag so later work can continue on a new version. Do not move or reuse a published tag. Report any correction in a new version.

## Image upload

The primary image package contains one `images.zip`, one global `metadata.csv`, README, licence and checksums. It preserves all 18,651 photograph files. The ZIP is about 59.7 GB. The user has allocated 70 GB to the image draft. The two verified year ZIPs are preserved under the local preparation folder as a fallback. They contain the same photos. Do not upload both packaging variants.

References: [Zenodo file limits](https://help.zenodo.org/docs/deposit/manage-files/) and [storage quota](https://help.zenodo.org/docs/deposit/manage-quota/).
