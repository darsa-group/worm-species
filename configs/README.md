# Configuration layout

Only the Genome paper-ablation configuration graph is retained.

```text
dev/genome_ablation_pipeline.yaml
├── configs/clusters/genome.yaml
├── dev/genome_ablation_baseline.yaml
│   └── config.yaml
│       └── configs/defaults/base.yaml
├── dev/genome_visual_ablation.yaml
│   └── dev/genome_ablation_baseline.yaml
├── dev/genome_data_holdouts.yaml
│   └── dev/genome_ablation_baseline.yaml
└── dev/paper_report_style.yaml

dev/genome_adult_taxon_ablation_pipeline.yaml
├── dev/genome_adult_taxon_baseline.yaml
├── dev/genome_adult_taxon_holdouts.yaml
└── dev/paper_report_style.yaml
```

The pipeline renders safely by default:

```bash
make ablation-pipeline
```

Explicit submission:

```bash
make ablation-pipeline PIPELINE_MODE=submit
```

The full plan is 175 fits: 45 baseline, 110 visual ablation, and 20 data
holdout. It also contains a shared base-cache job, a 21-task deterministic
condition-cache array, completed-result collectors, and a final paper-report
job.

The standalone Adult taxon plan contains 162 fits: 18 matching full-data
controls plus 144 holdouts spanning all eight observed Adult genus-species
combinations, three backbones, seeds 40/41/42, and hierarchy loss 0 and 0.2.
Render it with `make adult-taxon-ablation-pipeline`; submit explicitly with
`ADULT_TAXON_PIPELINE_MODE=submit`.
