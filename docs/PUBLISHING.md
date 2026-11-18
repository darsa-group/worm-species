# Coordinated publication

The source code and the two Zenodo records should be published in one release session. The source archive contains no images, model weights or generated research results. The companion dataset and model files are prepared separately.

1. Use `https://github.com/darsa-group/worm-species` and the planned version tag `v1.0.1`. The source is integrated into the existing repository root. Follow `docs/RELEASE_FREEZE.md` to commit the selected files without adding research outputs.
2. The reserved dataset DOI is `10.5281/zenodo.24341462` and the model DOI is `10.5281/zenodo.24341741`. Both DOIs and the versioned GitHub URL are in `configs/release_inputs.json` and the release notes. Enter creators, affiliations and ORCIDs yourself. Keep the image and checkpoint hashes unchanged.
3. Confirm the licence scopes. Original code uses MIT. Retain the Tabler icon, torchvision and Ultralytics notices and applicable source obligations. The segmentation model's upstream terms are not replaced by MIT or CC BY.
4. Run `make release-check` against the downloaded records and the short inference commands in the README.
5. Publish the GitHub release and both Zenodo records, add both records to the community, and check every public cross-link. Enter the final record URLs in the manuscript availability statements.

Creators remain for the user; both reserved DOIs were supplied by the user. A DOI reserved for a dataset is distinct from a paper DOI. The repository URL was read from its configured remote and the branch tips were checked without fetching. The local preparation performs no GitHub push, Zenodo upload or publication.
