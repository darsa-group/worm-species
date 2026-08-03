# Task-specific generalisation audit

Audit snapshot: branch `feat/task-specific-multitask-generalisation`, commit
`8e6d9c5`, inspected 2026-07-31. This document distinguishes three kinds of
evidence:

- **Confirmed by implementation/configuration** means the current code and fully
  resolved submission plans establish the statement.
- **Measured locally** means I additionally inspected the checked-in split CSVs
  or instantiated the model without downloading weights.
- **Not verifiable from available runs** means this repository contains no
  completed run artifacts for this experiment family. In particular, there are
  no family `config.json`, `history.csv`, `best_model.pt`,
  `test_metrics_best.json`, or row-level prediction files to compare.

The resolved common scientific settings are: `convnext_base`, pretrained and
unfrozen; 200 epochs; batch size 256; learning rate 0.0003; weight decay 0.0001;
AMP enabled; six workers; validation every five epochs; early-stopping patience
six and minimum delta 0.0001; task weights genus/species/age = 1.0/0.5/2.0;
hierarchy loss disabled; predefined splits enabled; and seeds 40, 41, 42 unless
a variant explicitly restricts the sweep to seed 42. The sources are
`configs/train/generalisation/_base.yaml:4-49,60-100`,
`dev/genome_ablation_baseline.yaml:9-32`, and
`configs/defaults/base.yaml:120-150`.

## Executive conclusion

The architecture mechanisms are real at runtime: the split model is larger than
the shared-head model, its final ConvNeXt stage is duplicated with no parameter
aliases, the joint sampler is installed in the training loader, PCGrad replaces
shared gradients, SupCon contributes to total loss, the adversary receives only
the age representation through gradient reversal, and single-task models have
one head only.

The current results still **must not be interpreted as individual-level,
fully reproducible scientific architecture evidence**. Holdout recall is an
image-row metric, repeated images from a biological individual are not
aggregated, predictions with individual identifiers are not saved, strict CUDA
determinism is not enabled, and no completed artifacts for these variants are
present locally. The valid code-level experiment design is therefore ahead of
the available audit trail.

## 1. Does every run use the same predefined train, validation, and test CSV?

**Yes for the generated Genome jobs; a direct local invocation is currently
misconfigured.** The inherited value is
`split.use_predefined_splits: true` and
`split.predefined_split_dir: split_csv`
(`configs/defaults/base.yaml:120-125`). The loader takes the predefined branch
instead of the seeded splitter (`src/worm_species/training/loaders.py:179-196`),
and the reader always appends
`split_csv/{train_split,val_split,test_split}.csv`
(`src/worm_species/data/labels.py:109-126`). The Genome array script corrects the
directory at runtime to the project root with
`split.predefined_split_dir="$PROJECT_ROOT"`
(`slurm/templates/node_shared_cache_array_job.sh.tmpl:162-178`), so every array
task reads these three checked-in files:

| split | rows | unique `barcode` | SHA-256 |
|---|---:|---:|---|
| train | 3,996 | 718 | `001d69ffb2216745bcc5c94609c6d3c5c3f2e629777c3c5763d101990599c84f` |
| validation | 921 | 166 | `12156d4a83643d5edda79c4978916713b95fb3b7df2628d407bc20d406d87226` |
| test | 1,274 | 221 | `832cd598fd835c9203959d4c508eb2d72cbd1236a65f10b6a4d73018a2da843f` |

For a direct local run, the inherited value `split_csv` is passed to a reader
that appends another `split_csv`, producing the nonexistent
`split_csv/split_csv/*.csv`. Local execution therefore needs
`--override split.predefined_split_dir=.` or the config/reader contract must be
fixed.

## 2. Are structured holdouts removed only from training and validation, with the test cohort unchanged?

**Yes.** Every structured definition explicitly says
`remove_from: [train, validation]`
(`configs/train/generalisation/_base.yaml:71-98`). The implementation makes
mutable copies of only those two frames, filters them, derives the independent
cohort from the original test frame, records `test_unchanged: true`, and returns
the original `test` object (`src/worm_species/data/holdouts.py:70-102,116-153`).
The loader then uses those returned frames (`src/worm_species/training/loaders.py:215-232`).

## 3. Are all images from one individual restricted to a single split?

**Yes, measured for the checked-in CSVs.** `data.group_col` resolves to
`barcode` (`configs/defaults/base.yaml:35-40`). The observed pairwise barcode
overlaps are train-validation = 0, train-test = 0, and validation-test = 0.
The split generator also operates on one row per group and asserts all three
sets are disjoint before materialising image rows
(`src/splits.py:36-52,54-92`). Because these experiments use the predefined
files, the measured zero-overlap audit—not a new seeded split—is the relevant
evidence.

## 4. Which code components are affected by the random seed?

The resolved seed affects:

- **Initialisation:** `run_one` seeds before loaders and model construction
  (`src/worm_species/training/runner.py:246-254,295-321`), and `set_seed` seeds
  Python, NumPy, CPU Torch, and all CUDA generators
  (`src/worm_species/training/reproducibility.py:11-16`). This covers new head,
  branch, projection, and adversary parameters.
- **Default sampling and dataloader order:** the training loader uses
  `shuffle=True` when no custom sampler exists, while validation and test never
  shuffle (`src/worm_species/training/loaders.py:394-440`). It relies on Torch's
  global generator; no explicit DataLoader generator is supplied.
- **Joint sampling:** `JointSpeciesStageSampler` receives `cfg.seed`
  (`src/worm_species/training/loaders.py:398-414`), seeds a private generator
  with `seed + epoch`, and draws combination, individual, then image
  (`src/worm_species/data/samplers.py:94-121`). The epoch loop calls
  `sampler.set_epoch(epoch)` (`src/worm_species/training/epochs.py:61-65`).
- **Augmentation:** train-only random horizontal flip, vertical flip, and
  rotation resolve to probabilities 0.5, 0.5, and 270 degrees
  (`config.yaml:18-23`; `src/worm_species/data/transforms.py:32-68,138-156`).
  Torchvision consumes worker Torch RNG state.
- **Dropout:** split task-attention dropout is 0.1 and adapter dropout is 0.2;
  the corresponding modules are created at
  `src/worm_species/models/multitask.py:89-110,113-130,380-397`. The shared-head
  global-average configuration creates no extra pooling dropout module.
- **Structured holdout membership:** not affected. It is a deterministic filter
  over fixed CSVs (`src/worm_species/data/holdouts.py:26-50,75-102`).

## 5. Is training fully deterministic for the same seed and configuration?

**No hard guarantee.** Seeding is broad but the repository does not call
`torch.use_deterministic_algorithms`, configure deterministic cuDNN/CUBLAS, or
provide explicit `DataLoader(generator=...)` and `worker_init_fn`. The entire
determinism implementation is only the four seed calls in
`src/worm_species/training/reproducibility.py:11-16`, while loaders use six
workers and pinned/prefetched batches (`src/worm_species/training/loaders.py:386-440`)
and training uses CUDA AMP (`src/worm_species/training/runner.py:378-390`). Runs
on the same software/hardware stack will often reproduce closely, but bitwise
identity is not established by this code.

## 6. Are `shared_heads` and `split_taxonomy_age` genuinely different architectures at runtime?

**Yes.** `build_multitask_model` dispatches them to `MultiTaskClassifier` and
`SplitTaxonomyAgeClassifier`, respectively
(`src/worm_species/models/multitask.py:545-597`). With the resolved
`convnext_base`, three genus classes, eight species classes, and two age
classes, local construction measured:

| architecture | resolved branch | parameters |
|---|---|---:|
| shared heads | one backbone plus three linear heads | 87,579,789 |
| split taxonomy/age | duplicated final ConvNeXt stage plus separate pooling/heads | 112,927,887 |

`branch_mode: auto` resolves to `duplicated_final_stage` for ConvNeXt because
the model has a separable final stage
(`src/worm_species/models/multitask.py:360-381,421-450`). These are not aliases
or cosmetic config names.

## 7. Are taxonomy and age branches separate parameter objects, with no shared parameter aliases?

**Yes for the branch-specific modules; the early backbone is intentionally
shared.** The taxonomy final stage is retained and the age final stage is made
with `copy.deepcopy` (`src/worm_species/models/multitask.py:433-450`). Separate
task-attention pools and task heads are constructed as separate modules
(`src/worm_species/models/multitask.py:391-410`). A local runtime identity audit
found zero duplicate parameter-object IDs overall and zero intersections for
taxonomy versus age final stage, taxonomy versus age pool, and taxonomy/species
heads versus age head. The test contract independently checks distinct objects,
storage pointers, and initially equal values
(`tests/test_generalisation_models.py:148-176`). The shared stem/early stages at
`src/worm_species/models/multitask.py:440-448` are deliberately one parameter
set and are what `shared_parameters()` returns for PCGrad
(`src/worm_species/models/multitask.py:533-536`).

## 8. Does each named variant enable its intended mechanism in the resolved config?

**Yes.** The 105-run plan contains no duplicate resolved configuration hashes,
and the resolved mechanism matrix is:

| variant | runs / seeds | architecture / target | sampler | gradients | SupCon | adversary | pooling |
|---|---|---|---|---|---|---|---|
| `shared_heads` | 15 / 40,41,42 | shared / all | default | standard | off | off | global average |
| `single_task_age` | 15 / 40,41,42 | single / age | default | standard | off | off | global average |
| `single_task_genus` | 5 / 42 | single / genus | default | standard | off | off | global average |
| `single_task_species` | 5 / 42 | single / species | default | standard | off | off | global average |
| `split_taxonomy_age` | 15 / 40,41,42 | split / all | default | standard | off | off | task attention |
| `split_joint_sampler` | 5 / 42 | split / all | joint | standard | off | off | task attention |
| `split_pcgrad` | 5 / 42 | split / all | default | PCGrad | off | off | task attention |
| `split_age_supcon` | 5 / 42 | split / all | default | standard | on, 0.1, T=0.07 | off | task attention |
| `split_joint_sampler_pcgrad` | 15 / 40,41,42 | split / all | joint | PCGrad | off | off | task attention |
| `split_full` | 15 / 40,41,42 | split / all | joint | PCGrad | on, 0.1, T=0.07 | off | task attention |
| `split_species_adversary` | 5 / 42 | split / all | default | standard | off | on, loss=0.05, max GRL=0.1 | task attention |

The direct definitions are in
`configs/train/generalisation/shared_heads.yaml:3-13`,
`single_task_age.yaml:3-19`, `single_task_genus.yaml:3-23`,
`single_task_species.yaml:3-23`, `split_taxonomy_age.yaml:3-16`,
`split_joint_sampler.yaml:3-19`, `split_pcgrad.yaml:3-16`,
`split_age_supcon.yaml:3-20`, `split_joint_sampler_pcgrad.yaml:3-18`,
`split_full.yaml:3-25`, and `split_species_adversary.yaml:3-20`, all under
`configs/train/generalisation/`. `split_full` intentionally does **not** include
the exploratory adversary.

## 9. Is `joint_species_stage` sampling instantiated in the training dataloader?

**Yes.** When the resolved type is `joint_species_stage`, the loader constructs
`JointSpeciesStageSampler` from the post-holdout training dataframe and assigns
it as `sampler`, with `shuffle=False` implied by `shuffle=train_sampler is None`
(`src/worm_species/training/loaders.py:394-428`). Validation, test, and holdout
loaders never receive it (`src/worm_species/training/loaders.py:429-454`). With
replacement and `samples_per_epoch: null`, one epoch has `len(train_df)` draws
(`src/worm_species/data/samplers.py:41-55`).

## 10. Is PCGrad actually used during optimisation of shared parameters?

**Yes.** `pcgrad` activates only while training, obtains the model's explicitly
shared parameters, calculates task gradients, projects conflicts, and replaces
only those shared gradients before `optimizer.step`/`scaler.step`
(`src/worm_species/training/epochs.py:78-100,278-351`). The projection and
auxiliary-gradient preservation are implemented at
`src/worm_species/training/gradients.py:90-166`. For the split ConvNeXt model,
the affected parameters are the shared early backbone only
(`src/worm_species/models/multitask.py:533-536`); task-specific final stages,
heads, SupCon projection, and adversary use their ordinary gradients.

## 11. Is the age supervised-contrastive loss calculated and added to total loss?

**Yes when enabled and when a batch has at least one valid positive pair.** The
split model creates a 256-to-128 age projection only when SupCon is enabled
(`src/worm_species/models/multitask.py:403-410,576-596`). The epoch loop computes
the loss from age embeddings, age labels, and optionally species labels, then
adds `weight * supcon_loss` to `total_loss`
(`src/worm_species/training/epochs.py:224-258`). The loss prefers cross-species
same-age positives and returns `None` if no valid anchors exist
(`src/worm_species/training/losses.py:199-265`). Thus it is active, but some
batches legitimately contribute no contrastive term; anchor counts are reported
at `src/worm_species/training/epochs.py:428-442`.

## 12. Is the species adversary active, and does gradient reversal affect only the age branch?

**Yes only in `split_species_adversary`.** Its resolved settings are enabled,
loss weight 0.05, ten-epoch warmup, and maximum reversal coefficient 0.1
(`configs/train/generalisation/split_species_adversary.yaml:3-12`). The adversary
is fed `gradient_reverse(age_features, coefficient)` and never taxonomy features
(`src/worm_species/models/multitask.py:499-530`). The coefficient ramps as
`0.1 * min(epoch/10, 1)` (`src/worm_species/training/epochs.py:28-35,118-123`),
and cross-entropy contributes `0.05 * adversary_loss`
(`src/worm_species/training/epochs.py:260-276`). Reversal affects the age branch
and the early backbone upstream of it; it does not flow through the taxonomy
final stage or taxonomy heads.

## 13. Do single-task models instantiate and optimise only their requested task head?

**Yes.** The loader reduces `target_cols` to exactly the configured target before
building label maps and criteria (`src/worm_species/training/loaders.py:234-260`).
`SingleTaskClassifier` passes only that target's class count to its parent and
therefore constructs one head (`src/worm_species/models/multitask.py:307-333`).
The optimizer receives only parameters belonging to that one-head model
(`src/worm_species/training/runner.py:314-349,378-382`). The backbone is still
optimised because `freeze_backbone: false`; “single task” means one supervised
head, not a frozen or head-only model. The focused test verifies absent logits
for other tasks (`tests/test_generalisation_models.py:136-146`).

## 14. Why do single-task runs report metrics for tasks they should not predict?

**The current trainer does not calculate non-target task metrics.** Its active
task list comes from `criteria.keys()` and it produces metrics only for those
tasks (`src/worm_species/training/epochs.py:69-72,135-136,462-500`). The focused
test explicitly verifies that an age-only epoch has no genus/species macro-F1
(`tests/test_generalisation_training.py:199-217`).

The generalisation collector does define a common row schema containing
`test_genus_macro_f1`, `test_species_macro_f1`, and `test_age_macro_f1`; missing
keys become `None`/NaN (`src/worm_species/analysis/generalisation_report.py:173-196`).
Therefore blank columns are expected. Any **non-null** metrics for unrequested
tasks cannot come from a current single-task run and indicate a stale/mixed run
directory, a legacy model, or an upstream display issue. No relevant completed
artifacts are present locally to identify which one occurred.

## 15. Are any run directories or metric files reused across differently named variants?

**No collision exists in the current planned 105 configurations, but rerun
protection is incomplete.** The planner prefixes outer run IDs by variant
(`src/worm_species/slurm/planning.py:218-262`), rejects duplicate config hashes
and output paths within a plan (`src/worm_species/slurm/planning.py:333-353,394-406`),
and the array script writes each run beneath `${RESULTS_ROOT}/${RUN_ID}`
(`slurm/templates/node_shared_cache_array_job.sh.tmpl:144-160`). Inside that
directory the trainer creates a second hash-based run directory
(`src/worm_species/training/runner.py:295-299`; naming algorithm at
`src/worm_species/training/naming.py:10-28`).

However, directories use `exist_ok=True` and files are overwritten in place. If
the same generated array task is manually rerun against the same `RUN_ID`, its
results can merge with/overwrite prior files; there is no completed-run refusal
or atomic fresh-directory requirement. This is a provenance risk, not evidence
that different current variants collide.

## 16. Are result families assigned from resolved configuration values or merely from run names?

**From resolved configuration values.** The report reads each run's
`config.json` and derives architecture, target, sampler, gradient strategy,
SupCon, and adversary fields (`src/worm_species/analysis/generalisation_report.py:87-128,137-180`).
Run-name prefixes are not used for family assignment. One limitation is that the
classification is precedence-based: any future split config with adversary plus
other mechanisms is labelled `split_species_adversary`, so the full mechanism
columns must also be retained when comparing compositions.

## 17. Are identical metrics exact at full precision, and do they come from identical prediction files?

**Not verifiable.** There are no completed generalisation metric artifacts in
the repository, and the current trainer does not write row-level prediction
files at all. It keeps `true` and `pred` in memory, writes aggregate JSON,
classification reports, and confusion matrices
(`src/worm_species/training/runner.py:84-174`), then discards the arrays.
Consequently, displayed equality cannot be checked at full precision or traced
to identical predictions. This is a missing provenance artifact, not proof of
duplication.

## 18. Are holdout recalls calculated per image or per biological individual?

**Per image row.** `run_hierarchy_epoch` appends one true/predicted class per
batch item (`src/worm_species/training/epochs.py:384-403`). Holdout recall counts
those array elements and averages correctness over the target mask
(`src/worm_species/evaluation/data_holdout.py:69-94`). No barcode enters the
calculation; the dataset returns image path and labels but not group ID
(`src/worm_species/data/datasets.py:78-137`).

## 19. How many unique individuals correspond to each reported holdout `n`?

The exact independent-test cohort sizes from the checked-in CSV are:

| holdout | reported task `n` (images) | unique individuals |
|---|---:|---:|
| `juvenile_aporrectodea_longa` | 163 for age and species | 29 |
| `juvenile_allolobophora_chlorotica` | 108 for age and species | 19 |
| `juvenile_genus_aporrectodea` | 600 for age and genus | 107 |
| `unseen_species_aporrectodea_longa_for_genus` | 191 for genus | 33 |

For reference, development removal is 455 images/84 individuals from train plus
98/18 from validation for juvenile A. longa; 421/76 plus 101/18 for juvenile A.
chlorotica; 1,901/350 plus 476/88 for juvenile genus Aporrectodea; and 588/106
plus 109/20 for unseen A. longa. The code records both rows and individuals in
its audit (`src/worm_species/data/holdouts.py:53-58,116-145`), but the task CSV's
`n` remains the image count (`src/worm_species/evaluation/data_holdout.py:81-94`).

## 20. How are multiple image predictions aggregated for one individual?

**They are not aggregated.** Every image contributes independently to metrics
and recall. There is no mean-probability, mean-logit, majority-vote, or
one-individual-one-vote operation in `run_hierarchy_epoch`
(`src/worm_species/training/epochs.py:141-165,384-500`) or
`evaluate_data_holdout` (`src/worm_species/evaluation/data_holdout.py:46-94`).

## 21. Which checkpoint is evaluated?

**Both are evaluated, but canonical/legacy test and all subsequent holdout
results use the best validation checkpoint.** The best checkpoint is selected
only from the configured validation metric and saved on improvement
(`src/worm_species/training/runner.py:392-403,497-559`). The final state is saved
as `last_model.pt`; last is evaluated first; best is then loaded and evaluated
with `write_legacy_outputs=True`, leaving the model on best weights for holdout
evaluation (`src/worm_species/training/runner.py:576-641,650-677`). Thus:

- `test_metrics_last.json` = final/early-stop state;
- `test_metrics_best.json` and legacy `test_metrics.json` = validation-selected
  `best_model.pt`;
- structured holdout metrics = `best_model.pt`;
- no checkpoint is selected using test performance.

## 22. Do different seeds select substantially different best epochs?

**Not verifiable from local artifacts.** `best_epoch` is recorded in each
checkpoint and run summary (`src/worm_species/training/checkpoints.py:32-54`;
`src/worm_species/training/runner.py:815-875`), but no completed histories or
checkpoints for this family are present. The configuration alone cannot answer
an empirical “substantially different” question.

## 23. Are the same holdout individuals evaluated for every architecture and seed?

**Yes by construction, assuming every job reads the three hashed CSVs above.**
Architecture and seed do not participate in the deterministic `evaluation_where`
filter (`src/worm_species/data/holdouts.py:75-102`), and all variants inherit the
same five holdout definitions (`configs/train/generalisation/_base.yaml:69-100`).
The exact test individual sets are therefore shared across matched holdout runs.
This should still be persisted and hash-checked per run; it currently is only
indirectly represented by counts in `split_summary.json`.

## 24. For each seed, which individual predictions change between architectures?

**Not answerable with current artifacts or output schema.** The repository has
no completed new-family runs, and even a completed run would not save barcode,
image identifier, probabilities, or per-row predictions. The dataset omits
barcode from returned samples (`src/worm_species/data/datasets.py:132-137`) and
the trainer discards in-memory predictions after aggregate reporting
(`src/worm_species/training/runner.py:84-185,621-648`).

## 25. Are class-label mappings identical across runs and loaded from the correct run directory?

**For matched architecture/seed runs with the same holdout, mappings are
deterministically derived from the same post-holdout training frame; complete
mapping dictionaries are intentionally different for single-task models and can
differ across holdouts.** Labels are sorted from training only
(`src/worm_species/data/labels.py:81-106`) and written inside each run directory
(`src/worm_species/training/runner.py:295-305`). Measured class counts are:

- original and the three juvenile-combination holdouts: genus 3, species 8,
  age 2;
- `unseen_species_aporrectodea_longa_for_genus`: genus 3, species 7, age 2,
  because A. longa is absent from training;
- single-task runs contain only their requested mapping.

The checkpoint also stores both mapping directions
(`src/worm_species/training/checkpoints.py:46-54`). During same-run evaluation,
`run_test_evaluation` loads only `model_state` and continues using the loader's
current maps (`src/worm_species/training/runner.py:80-110`), so it is correct in
the normal one-run path. It does **not** assert that checkpoint maps equal loader
maps; cross-directory checkpoint reuse would therefore need an additional
fail-closed mapping check. With no completed family runs, byte-for-byte mapping
identity cannot be audited empirically.

## 26. Verification table

The following is the strongest table possible from the repository today. Each
row is the seed-42, `original_baseline` representative from the fully resolved
plan. Config hashes use the planner's canonical YAML SHA-256 algorithm
(`src/worm_species/slurm/planning.py:133-135,375-391`). Parameter counts were
measured with the resolved class counts; pretrained weights do not alter counts.
The split-manifest hash is an audit SHA-256 over the three file hashes above:
`d2fab9bf1eab2ed7ea0407bc74c456257472636165fe0eb6274affff838b13c7`.

| planned run family | effective mechanism summary | config SHA-256 | parameters | split hash | checkpoint hash | prediction hash | metric hash |
|---|---|---|---:|---|---|---|---|
| `shared_heads` | shared, default, standard | `55da10fecbc3445dff9e5a69f8514bdeaedfb2c5e01964382cdad74ac5e0a8ea` | 87,579,789 | `d2fab9…13c7` | missing | not emitted | missing |
| `single_task_age` | single age | `d4322c2540d01694709a64ac4e7816ae7b3be4de823e2a730ae952766c5c476e` | 87,568,514 | `d2fab9…13c7` | missing | not emitted | missing |
| `single_task_genus` | single genus | `10fba7ee023d4dfb38dd0e618a26bb17172a9abf5d9fbe3010c2e15ad6f3a568` | 87,569,539 | `d2fab9…13c7` | missing | not emitted | missing |
| `single_task_species` | single species | `0faca56e77e6c19b89316522282777be0d10946efc97d8e627b02c14a0cb14d6` | 87,574,664 | `d2fab9…13c7` | missing | not emitted | missing |
| `split_taxonomy_age` | split, default, standard | `4e973f60023136839c94fd2a46ba53034003f0b976edc0578d455d840c35b43a` | 112,927,887 | `d2fab9…13c7` | missing | not emitted | missing |
| `split_joint_sampler` | split + joint | `c13b4c4d3e4efdb3d687b2d7f8b55b370b62f3e89c206b982e5550344bcd7abc` | 112,927,887 | `d2fab9…13c7` | missing | not emitted | missing |
| `split_pcgrad` | split + PCGrad | `4229cc7863028245be51d9a94ca0a629a4dd768b9ccc25981298433d9f0d336e` | 112,927,887 | `d2fab9…13c7` | missing | not emitted | missing |
| `split_age_supcon` | split + SupCon | `463b3c0cf843d504bb10ab28d0e597fb992133e93f400b8ebb454bc18c25e874` | 113,223,183 | `d2fab9…13c7` | missing | not emitted | missing |
| `split_joint_sampler_pcgrad` | split + joint + PCGrad | `463e2a4cf4113d6e7b9155b24951cb97a4b68a57b677c881594c43d2a94fe5b1` | 112,927,887 | `d2fab9…13c7` | missing | not emitted | missing |
| `split_full` | split + joint + PCGrad + SupCon | `ec7d24a7d94dc9763f8700c3573713548b852a2b2ebcf67015a40d46d5c1bdf0` | 113,223,183 | `d2fab9…13c7` | missing | not emitted | missing |
| `split_species_adversary` | split + adversary | `523eb24cd329dccc36b49bcab24058131f886e23ce10f7b83bdfdf3109986238` | 112,936,087 | `d2fab9…13c7` | missing | not emitted | missing |

A complete 105-row table with actual outer run ID, effective saved config hash,
parameter count, split hash, selected checkpoint path/hash, prediction hash, and
metric hash can be generated only after the runs and prediction-ledger change
described below exist. The current `model_parameters.json` output already
provides parameter counts (`src/worm_species/training/runner.py:327-340`).

## 27. Which comparisons are valid, duplicated, mislabelled, or unsupported?

### Valid by design, once complete artifacts are available

- `shared_heads` versus `split_taxonomy_age`, paired by holdout and seeds
  40/41/42, isolates the architecture/pooling package.
- `single_task_age` versus shared/split age performance, paired by holdout and
  seeds 40/41/42, addresses negative transfer versus multitask learning.
- At seed 42, `split_taxonomy_age` versus each of
  `split_joint_sampler`, `split_pcgrad`, `split_age_supcon`, and
  `split_species_adversary` isolates one mechanism at a time.
- `split_joint_sampler_pcgrad` versus `split_taxonomy_age`, paired across all
  three seeds, estimates the joint two-mechanism package.
- `split_full` versus `split_taxonomy_age`, paired across all three seeds,
  estimates the joint sampler + PCGrad + SupCon package.
- Every structured-holdout comparison is valid only when architecture, seed,
  and holdout name are matched; the fixed test cohort is then the same.

### Duplicated or mislabelled

- **No duplicate resolved configurations** exist in the 105-run plans. In
  particular, `split_joint_sampler_pcgrad` and `split_full` differ because only
  the latter enables SupCon.
- **No current family is mislabelled by resolved config.** Family assignment is
  config-derived (`src/worm_species/analysis/generalisation_report.py:87-128`).
- Common-schema blank/NaN single-task columns are not predictions and must not
  be presented as measured task results.

### Unsupported or overclaimed

- Multi-seed uncertainty claims for genus-only, species-only, joint-only,
  PCGrad-only, SupCon-only, or adversary-only variants: each has only seed 42.
- Attribution of the `split_full` gain to any one mechanism; it changes three
  mechanisms simultaneously.
- Attribution of the three-seed joint-sampler+PCGrad result to sampler or
  PCGrad individually; the corresponding isolated arms are single-seed.
- Treating the adversary as part of `split_full`; it is explicitly disabled
  there by inherited config (`configs/train/generalisation/_base.yaml:14-18` and
  `split_full.yaml:1-25`).
- Comparisons across different structured holdout names, because they use
  different image and individual cohorts.
- Any individual-level accuracy/recall claim, because current metrics are
  image-level.
- Any claim of bitwise reproducibility, exact prediction duplication,
  seed-specific best epochs, or architecture-specific changed individuals,
  because the required guarantees/artifacts are absent.
- Any numerical architecture conclusion from this repository snapshot: no
  completed new-family run artifacts are locally available.

## 28. Required code changes before scientific interpretation

The following are required, in priority order:

1. **Save an immutable row-level prediction ledger for best and last
   checkpoints.** Include run ID, checkpoint hash, split/cohort, image path or
   stable row ID, barcode, true label, predicted label, logits/probabilities,
   and class-map hash for every task. This requires returning group/row identity
   from `MultiTaskWormImageDataset` (`src/worm_species/data/datasets.py:78-137`)
   and writing the arrays currently discarded by
   `run_test_evaluation` (`src/worm_species/training/runner.py:84-185`).
2. **Define and report individual-level evaluation.** Aggregate all images for
   one barcode with a preregistered rule—recommended: mean class probability,
   then argmax—and report both image-level and one-individual-one-vote metrics.
   Holdout `n` must expose `n_images` and `n_individuals`; confidence intervals
   or bootstrap resampling must cluster by individual.
3. **Persist a provenance manifest per run.** Record Git commit, fully effective
   post-CLI config and SHA-256, all three split file hashes, exact cohort barcode
   hashes, class-map hash, parameter count, selected best epoch/validation score,
   best/last checkpoint hashes, prediction-ledger hashes, metric hashes, and
   Python/PyTorch/torchvision/CUDA/cuDNN versions. The planner already computes a
   config hash (`src/worm_species/slurm/planning.py:133-135`), but the trainer
   output does not tie all artifacts together.
4. **Fail closed on output reuse.** Refuse to start in a non-empty completed run
   directory unless an explicit resume mode verifies config, split, class-map,
   and checkpoint hashes. Write results atomically and publish a completion
   manifest last. Current `exist_ok=True` and in-place writes are at
   `src/worm_species/training/runner.py:295-305`.
5. **Add a strict determinism mode.** Configure deterministic Torch algorithms,
   cuDNN and CUBLAS requirements, explicit train/eval DataLoader generators,
   deterministic worker seeding, and record any operation that cannot comply.
   The current seed-only implementation at
   `src/worm_species/training/reproducibility.py:11-16` is insufficient.
6. **Validate checkpoint identity before evaluation.** Assert checkpoint config,
   label maps, parameter architecture/count, split hashes, and current run
   directory all agree before loading. At present only `model_state` is consumed
   (`src/worm_species/training/runner.py:80-83`).
7. **Make reporting completeness-aware.** Require the expected paired
   architecture/seed/holdout grid, reject duplicate config or prediction hashes,
   distinguish missing metrics from zero, and refuse paired tests when seeds or
   cohort hashes do not match. Continue deriving families from configs, not
   names (`src/worm_species/analysis/generalisation_report.py:87-128`).
8. **Fix the direct-run split-directory contract.** Either set
   `predefined_split_dir: .` or make `read_csvs_from_dir` accept the actual
   `split_csv` directory without appending it again
   (`configs/defaults/base.yaml:120-125` versus
   `src/worm_species/data/labels.py:109-126`).
9. **Rerun the full matched design after these changes.** Existing results, if
   stored externally, cannot be retrofitted with missing individual predictions
   and provenance. Architecture claims should use paired seeds and identical
   cohort hashes, and single-seed mechanism arms should remain explicitly
   exploratory until expanded.

Until items 1-7 are implemented and the matched runs are regenerated, the code
supports statements about mechanism implementation, but not strong scientific
statements about per-individual biological generalisation or exact differences
between architectures.
