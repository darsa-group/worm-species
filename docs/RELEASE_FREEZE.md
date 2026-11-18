# Release 1.0.1

The paper branch is `paper-pipeline-cache-only` in `darsa-group/worm-species`. Version `v1.0.1` contains the runnable source, required configurations, figure assets and documentation. Earlier research folders and generated results are kept locally and excluded from Git. The previous `v1.0.0` tag is retained in the repository history.

The dataset DOI is `10.5281/zenodo.24341462`. The model DOI is `10.5281/zenodo.24341741`. Both are recorded in the README, configuration and release notes. The code release URL is https://github.com/darsa-group/worm-species/releases/tag/v1.0.1.

`configs/release_source_paths.txt` lists the packaged source. `SOURCE_MANIFEST.csv` and `SHA256SUMS` record its integrity. To prepare a later edit, build a fresh archive, review the selected diff and commit the changed source and manifests:

```bash
python scripts/package_source.py --output work/source-rebuilt.zip
python scripts/check_release_freeze.py --require-links
git add --pathspec-from-file=configs/release_commit_paths.txt
git diff --cached --check
git diff --cached --stat
git commit -m "Describe the source change"
make release-freeze-check
```

Create a new version for changes after publication. Keep published tags fixed. The source archive contains no photographs or model weights; those belong in the two Zenodo records. Creators, affiliations and ORCIDs are entered on Zenodo by the user.
