# Configuration layout

The Genome paper-ablation graph remains the paper workflow. Task-specific
multitask diagnostics are an additional, isolated configuration family.

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
│   └── dev/genome_ablation_baseline.yaml
├── dev/genome_adult_taxon_holdouts.yaml
│   └── dev/genome_adult_taxon_baseline.yaml
└── dev/paper_report_style.yaml

configs/train/generalisation/_base.yaml
├── shared_heads.yaml
├── single_task_genus.yaml
├── single_task_species.yaml
├── single_task_age.yaml
├── split_taxonomy_age.yaml
│   ├── split_joint_sampler.yaml
│   ├── split_pcgrad.yaml
│   ├── split_age_supcon.yaml
│   ├── split_joint_sampler_pcgrad.yaml
│   ├── split_full.yaml
│   └── split_species_adversary.yaml
```

The pipeline renders safely by default:

```bash
make ablation-pipeline
```

Explicit submission:

```bash
make ablation-pipeline PIPELINE_MODE=submit
```

The paper pipeline contains 1,470 fits plus shared cache, collector, and report
jobs. Generalisation configs are submitted separately to avoid accidental
Cartesian expansion. The five principal matrices each contain 15 runs: three
seeds across the original split and four structured holdouts. Isolated
mechanism and exploratory matrices contain five seed-42 runs unless explicitly
promoted.

The standalone Adult taxon-ablation pipeline contains 270 fits: five
backbones, seeds 40/41/42, hierarchy loss 0 and 0.2, 30 full-data controls,
and 240 fits spanning the eight observed Adult genus-species combinations.
Render it with `make adult-taxon-ablation-pipeline`; submission requires
`ADULT_TAXON_PIPELINE_MODE=submit`.

Validate one diagnostic matrix with:

```bash
make generalisation-validate \
  GENERALISATION_CONFIG=configs/train/generalisation/split_full.yaml
```
