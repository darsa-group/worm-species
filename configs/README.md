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

The full plan is 1,470 fits: 90 baseline, 660 visual ablation, 600 visual
interaction, and 120 data-holdout fits. It also contains a shared base-cache
job, a deterministic condition-cache array, completed-result collectors, and
a final paper-report job.

The standalone taxon-stage plan contains 360 fits: 30 matching full-data
controls plus 330 holdouts spanning eight Adult and three Juvenile
genus-species-stage combinations, five backbones, seeds 40/41/42, and
hierarchy loss 0 and 0.2.
Render it with `make adult-taxon-ablation-pipeline`; submit explicitly with
`ADULT_TAXON_PIPELINE_MODE=submit`.
