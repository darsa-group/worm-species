# Repository inventory

Repository root: `/home/devd/worm-species`

This is a static inventory. It does not prove runtime equivalence or that every dynamically constructed path/import has been detected.

## Summary

| Category | Files |
| --- | --- |
| config | 6 |
| documentation | 11 |
| notebook | 19 |
| other | 7 |
| python | 37 |
| python_test | 1 |
| shell_or_slurm | 13 |

| Manifest source | Files |
| --- | --- |
| Git tracked | 53 |
| Targeted untracked source search | 41 |

### Extensions

| Extension | Files |
| --- | --- |
| .csv | 3 |
| .ipynb | 19 |
| .json | 2 |
| .md | 10 |
| .py | 38 |
| .sh | 13 |
| .tsv | 3 |
| .txt | 1 |
| .yaml | 4 |
| <none> | 1 |

## Excluded directories

| Path | Reason | Tracked files | Tracked bytes (lower bound) | Size note |
| --- | --- | --- | --- | --- |
| .git | generated/data-heavy directory name or prefix | 0 | 0 | Tracked-byte lower bound only; contents were not traversed. |
| .ipynb_checkpoints | generated/data-heavy directory name or prefix | 0 | 0 | Tracked-byte lower bound only; contents were not traversed. |
| .mypy_cache | generated/data-heavy directory name or prefix | 0 | 0 | Tracked-byte lower bound only; contents were not traversed. |
| .vscode | generated/data-heavy directory name or prefix | 0 | 0 | Tracked-byte lower bound only; contents were not traversed. |
| __pycache__ | generated/data-heavy directory name or prefix | 0 | 0 | Tracked-byte lower bound only; contents were not traversed. |
| outputs_slurm | generated/data-heavy directory name or prefix | 90 | 277949 | Tracked-byte lower bound only; contents were not traversed. |
| refactor_audit | generated/data-heavy directory name or prefix | 0 | 0 | Tracked-byte lower bound only; contents were not traversed. |
| refactor_audit_after | generated/data-heavy directory name or prefix | 0 | 0 | Tracked-byte lower bound only; contents were not traversed. |
| refactor_audit_bounded | generated/data-heavy directory name or prefix | 0 | 0 | Tracked-byte lower bound only; contents were not traversed. |
| single_task/outputs | generated/data-heavy directory name or prefix | 1 | 15561 | Tracked-byte lower bound only; contents were not traversed. |
| src/__pycache__ | generated/data-heavy directory name or prefix | 0 | 0 | Tracked-byte lower bound only; contents were not traversed. |
| src/worm_species/__pycache__ | generated/data-heavy directory name or prefix | 0 | 0 | Tracked-byte lower bound only; contents were not traversed. |
| src/worm_species/config/__pycache__ | generated/data-heavy directory name or prefix | 0 | 0 | Tracked-byte lower bound only; contents were not traversed. |
| src/worm_species/data/__pycache__ | generated/data-heavy directory name or prefix | 0 | 0 | Tracked-byte lower bound only; contents were not traversed. |
| src/worm_species/experiments/__pycache__ | generated/data-heavy directory name or prefix | 0 | 0 | Tracked-byte lower bound only; contents were not traversed. |
| src/worm_species/models/__pycache__ | generated/data-heavy directory name or prefix | 0 | 0 | Tracked-byte lower bound only; contents were not traversed. |

## Candidate entry points

| Path | Category | Main guard | Executable | CLI flags | Python calls | SBATCH directives |
| --- | --- | --- | --- | --- | --- | --- |
| .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/collect_dual_cue_results.py | python | True | False |  |  |  |
| .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/generate_dual_cue_run_specs.py | python | True | False |  |  |  |
| .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/submit_dual_cue_experiment.sh | shell_or_slurm | False | False |  | python "${PROJECT_SRC}/${RUN_SPEC_GENERATOR}" \<br>python - <<'PY'<br>python - "$BASE_CONFIG" "$SCRATCH_DATA" "$CACHE_ROOT" "${OVERRIDE_ARGS[@]}" <<'PY'<br>srun python "$TRAIN_SCRIPT" \<br>python "${PROJECT_SRC}/${RESULT_COLLECTOR}" "$RESULTS_ROOT" | #SBATCH -N 1<br>#SBATCH -n 1<br>#SBATCH -N 1<br>#SBATCH -n 1<br>#SBATCH --gres=gpu:1<br>#SBATCH -N 1<br>#SBATCH -n 1<br>#SBATCH -N 1 |
| .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/train_multitask_cue_suppression.py | python | True | False | --config, --override, --sweep |  |  |
| .codex/codex_repository_refactor_handoff/references/known_files/original/submit_node_local_sweep.sh | shell_or_slurm | False | False |  | python - "${PROJECT_SRC}/${BASE_CONFIG}" "$RUN_SPECS_DIR" "${RESULTS_ROOT}/sweep_plan.tsv" <<'PY'<br>python - <<'PY'<br>python - "$BASE_CONFIG" "$SCRATCH_DATA" "$CACHE_ROOT" "${OVERRIDE_ARGS[@]}" <<'PY'<br>srun python "$TRAIN_SCRIPT" \ | #SBATCH -N 1<br>#SBATCH -n 1<br>#SBATCH -N 1<br>#SBATCH -n 1<br>#SBATCH --gres=gpu:1<br>#SBATCH -N 1<br>#SBATCH -n 1 |
| .codex/codex_repository_refactor_handoff/references/known_files/original/train_multitask_masked_hloss.py | python | True | False | --config, --override, --sweep |  |  |
| .codex/codex_repository_refactor_handoff/tools/audit_repository.py | python | True | True | --ignore, --output-dir |  |  |
| .codex/codex_repository_refactor_handoff/tools/compare_contract_trees.py | python | True | True | --allow-extra |  |  |
| 01_build_persistent_cache_resolved.sh | shell_or_slurm | False | True |  | python --version<br>python - \ | #SBATCH --account=worm-species<br>#SBATCH --job-name=worm_build_cache<br>#SBATCH --cpus-per-task=8<br>#SBATCH --mem=16G<br>#SBATCH --time=02:00:00<br>#SBATCH --output=worm_build_cache_%j.out<br>#SBATCH --error=worm_build_cache_%j.err |
| 02_submit_sweep_cache_to_tmp_resolved.sh | shell_or_slurm | False | True |  | python - "${SOURCE_ROOT}/${BASE_CONFIG}" "$RUN_SPECS_DIR" "${RESULTS_ROOT}/sweep_plan.tsv" <<'PY'<br>python --version<br>python - <<'PY'<br>srun python "$TRAIN_SCRIPT" \ | #SBATCH --nodes=1<br>#SBATCH --ntasks=1<br>#SBATCH --gres=gpu:1 |
| archive_project_to_drive.sh | shell_or_slurm | False | True |  |  |  |
| collect_dual_cue_results.py | python | True | False |  |  |  |
| generate_dual_cue_run_specs.py | python | True | False |  |  |  |
| run_persistent_cache_sweep_wandb.sh | shell_or_slurm | False | False |  | python - "${SOURCE_ROOT}/${BASE_CONFIG}" "$RUN_SPECS_DIR" "${RESULTS_ROOT}/sweep_plan.tsv" <<'PY'<br>python --version<br>python - <<'PY'<br>srun python "$TRAIN_SCRIPT" \ | #SBATCH --nodes=1<br>#SBATCH --ntasks=1<br>#SBATCH --gres=gpu:1 |
| src/download_pretrained_from_config.py | python | True | False | --config, --torch-home |  |  |
| src/generate_sweep_run_specs.py | python | True | True | --config, --out-dir |  |  |
| submit_colour_ablation_sweep.sh | shell_or_slurm | False | False |  | python - "${PROJECT_SRC}/${BASE_CONFIG}" "$RUN_SPECS_DIR" "${RESULTS_ROOT}/sweep_plan.tsv" <<'PY'<br>python - <<'PY'<br>python - "$BASE_CONFIG" "$SCRATCH_DATA" "$CACHE_ROOT" "${OVERRIDE_ARGS[@]}" <<'PY'<br>srun python "$TRAIN_SCRIPT" \<br>python - "$RESULTS_ROOT" <<'PY' | #SBATCH -N 1<br>#SBATCH -n 1<br>#SBATCH -N 1<br>#SBATCH -n 1<br>#SBATCH --gres=gpu:1<br>#SBATCH -N 1<br>#SBATCH -n 1<br>#SBATCH -N 1 |
| submit_dual_cue_experiment.sh | shell_or_slurm | False | False |  | python "${PROJECT_SRC}/${RUN_SPEC_GENERATOR}" \<br>python - <<'PY'<br>python - "$BASE_CONFIG" "$SCRATCH_DATA" "$CACHE_ROOT" "${OVERRIDE_ARGS[@]}" <<'PY'<br>srun python "$TRAIN_SCRIPT" \<br>python "${PROJECT_SRC}/${RESULT_COLLECTOR}" "$RESULTS_ROOT" | #SBATCH -N 1<br>#SBATCH -n 1<br>#SBATCH -N 1<br>#SBATCH -n 1<br>#SBATCH --gres=gpu:1<br>#SBATCH -N 1<br>#SBATCH -n 1<br>#SBATCH -N 1 |
| submit_dual_cue_experiment_genome.sh | shell_or_slurm | False | False |  | python "${PROJECT_SRC}/${RUN_SPEC_GENERATOR}" \<br>python - <<'PY'<br>srun python "$TRAIN_SCRIPT" \<br>python "${PROJECT_SRC}/${RESULT_COLLECTOR}" "$RESULTS_ROOT" | #SBATCH -N 1<br>#SBATCH -n 1<br>#SBATCH --gres=gpu:1<br>#SBATCH -N 1<br>#SBATCH -n 1 |
| submit_worm_node_local_scratch_sweep.sh | shell_or_slurm | False | True |  | python - "${PROJECT_SRC}/${BASE_CONFIG}" "$RUN_SPECS_DIR" "${RESULTS_ROOT}/sweep_plan.tsv" <<'PY'<br>python - <<'PY'<br>python - "$BASE_CONFIG" "$SCRATCH_DATA" "$CACHE_ROOT" "${OVERRIDE_ARGS[@]}" <<'PY'<br>srun python "$TRAIN_SCRIPT" \ | #SBATCH -account worm-species<br>#SBATCH -N 1<br>#SBATCH -n 1<br>#SBATCH -account worm-species<br>#SBATCH -N 1<br>#SBATCH -n 1<br>#SBATCH --gres=gpu:1<br>#SBATCH -account worm-species |
| submit_worm_node_local_scratch_sweep_hloss.sh | shell_or_slurm | False | True |  | python - "${PROJECT_SRC}/${BASE_CONFIG}" "$RUN_SPECS_DIR" "${RESULTS_ROOT}/sweep_plan.tsv" <<'PY'<br>python - <<'PY'<br>python - "$BASE_CONFIG" "$SCRATCH_DATA" "$CACHE_ROOT" "${OVERRIDE_ARGS[@]}" <<'PY'<br>srun python "$TRAIN_SCRIPT" \ | #SBATCH -account worm-species<br>#SBATCH -N 1<br>#SBATCH -n 1<br>#SBATCH -account worm-species<br>#SBATCH -N 1<br>#SBATCH -n 1<br>#SBATCH --gres=gpu:1<br>#SBATCH -account worm-species |
| tests/test_behaviour_contracts.py | python_test | True | False |  |  |  |
| train_multitask_colour_ablation.py | python | True | False | --config, --override, --sweep |  |  |
| train_multitask_cue_suppression.py | python | True | False | --config, --override, --sweep |  |  |
| train_multitask_masked.py | python | True | False | --config, --override, --sweep |  |  |
| train_multitask_masked_hloss.py | python | True | False | --config, --override, --sweep |  |  |
| train_multitask_masked_hloss_wandb.py | python | True | False | --config, --override, --sweep |  |  |
| tranfser_from_genome.sh | shell_or_slurm | False | True |  |  |  |
| tranfser_from_ghpc.sh | shell_or_slurm | False | True |  |  |  |

## Notebooks

| Path | Code cells | Markdown cells | Imports | Referenced files | Magics/shell |
| --- | --- | --- | --- | --- | --- |
| cam_mutlitask.ipynb | None | None |  |  |  |
| data_leakage_check.ipynb | 21 | 14 | IPython.display, json, matplotlib.pyplot, numpy, pandas, pathlib, re, shlex, textwrap, warnings | all_sweep_results_aggregated.csv, best_model.pt, classification_report_{task}.csv, config.json, confusion_matrix_{task}.csv, history.csv, label_to_index_by_task.json, run_*/multi_run_results.csv, test_metrics.json, top_models_by_{RANK_BY}.csv, top_run_per_model_type_by_{RANK_BY}.csv |  |
| data_notebooks/data_analysis.ipynb | 9 | 0 | concurrent.futures, cv2, matplotlib.pyplot, numpy, os, pandas, pathlib, sklearn.ensemble, sklearn.impute, sklearn.metrics, sklearn.model_selection, sklearn.pipeline, sklearn.preprocessing, sklearn.utils.class_weight, tqdm.notebook | global_metadata.csv |  |
| data_notebooks/data_explore.ipynb | 6 | 0 | PIL, matplotlib.pyplot, os, pandas, random | global_metadata.csv |  |
| dataset_tes.ipynb | 12 | 0 | PIL, matplotlib.pyplot, numpy, pandas, pathlib, random, src.cache, src.dataset, src.dataset_multitask, src.splits, src.utils, sys, torch, torch.utils.data | ../petridish-worm-images/01_Segmented/global_metadata.csv, config.yaml |  |
| earthworm_colour_robustness_analysis.ipynb | 17 | 18 | IPython.display, __future__, matplotlib.pyplot, numpy, pandas, pathlib, re, typing | colour_robustness_summary.csv, manuscript_results_table.csv, matched_condition_macro_f1_long.csv, matched_condition_results.csv, matched_vs_rgb_stress_test.csv, rgb_model_cue_suppression_macro_f1_ratios.csv, rgb_model_cue_suppression_test_metrics.csv, rgb_model_cue_suppression_transform_summary.csv, saturation_curve_auc.csv |  |
| earthworm_cue_suppression_analysis_v2.ipynb | 24 | 16 | IPython.display, PIL, __future__, cv2, json, matplotlib.container, matplotlib.pyplot, numpy, pandas, pathlib, re, typing | *.csv, *.json, adaptation_gain_bootstrap_intervals.csv, architecture_difference_saturation.csv, condition_manifest.json, conditions.json, matched_condition_macro_f1_long.csv, matched_condition_manifest.json, matched_vs_rgb_stress_test.csv, non_saturation_relative_performance.csv, rgb_model_cue_suppression_macro_f1_ratios.csv, saturation_auc_summary.csv, saturation_endpoint_summary.csv, saturation_manuscript_table.csv, sweep_manifest.json |  |
| gradcam_multitask_all_tasks.ipynb | 18 | 11 | PIL, __future__, copy, json, math, matplotlib.cm, matplotlib.pyplot, numpy, pandas, pathlib, src.dataset_multitask, src.models, torch, torch.nn, torch.nn.functional | /mnt/extssd/Earthworms/petridish-worm-images/01_Segmented/global_metadata.csv, best_model.pt, config.json, label_to_index_by_task.json |  |
| single_task/analyze_species_outputs_v2.ipynb | 18 | 13 | IPython.display, json, matplotlib.pyplot, numpy, pandas, pathlib, re, typing | No early_stopping section found in config.json, all_runs_summary.csv, best_model.pt, classification_report.csv, config.json, confusion_matrix.csv, history.csv, label_to_index.json, ranked_runs.csv, ranked_runs_filtered.csv, split_summary.json, test_metrics.json, {safe}.csv |  |
| single_task/cam.ipynb | 14 | 0 | PIL, cv2, json, matplotlib.cm, matplotlib.pyplot, numpy, pandas, pathlib, src.dataset, torch, torch.nn, torch.nn.functional, torchvision | best_model.pt, config.json, label_to_index.json |  |
| single_task/umap.ipynb | 0 | 0 |  |  |  |
| single_task/umap_species_embeddings.ipynb | 14 | 12 | PIL, json, matplotlib.pyplot, numpy, pandas, pathlib, plotly.express, plotly.graph_objects, src.dataset, torch, torch.nn, torch.utils.data, torchvision, tqdm.auto, typing, umap | best_model.pt, config.json, embedding_metadata_{suffix}.csv, label_to_index.json, umap_embedding_table_{suffix}.csv |  |
| split_csv/check_splits copy.ipynb | 1 | 1 | pandas, pathlib | test_split.csv, train_split.csv, val_split.csv |  |
| worm_advanced_test_diagnostics_top_models.ipynb | 26 | 12 | IPython.display, PIL, __future__, math, matplotlib.pyplot, numpy, pandas, pathlib, re, sklearn.metrics, tqdm.auto, warnings | accuracy_by_image_feature_bins.csv, accuracy_by_parent_folder_min10.csv, aggregation_comparison_scores.csv, all_top_model_test_set_image_predictions_long.csv, bootstrap_ci_by_individual.csv, calibration_bins_all_models_tasks.csv, calibration_summary_ece.csv, individual_prediction_stability.csv, individual_predictions_all_aggregation_methods.csv, model_family_agreement_categories_image_level.csv, model_family_agreement_summary_image_level.csv, species_age_interaction_image_level.csv, stability_by_true_label.csv, test_image_quality_features.csv, test_predictions_with_image_quality_features.csv |  |
| worm_gradcam_top_model_families.ipynb | 15 | 9 | IPython.display, PIL, __future__, copy, dataclasses, json, math, matplotlib, matplotlib.cm, matplotlib.colors, matplotlib.pyplot, numpy, pandas, pathlib, re, shlex, src.dataset_multitask, src.models, src.splits, textwrap, torch, torch.nn, torch.nn.functional, warnings | /mnt/extssd/Earthworms/petridish-worm-images/01_Segmented/global_metadata.csv, best_model.pt, classification_report_{task}.csv, config.json, confusion_matrix_{task}.csv, history.csv, label_to_index_by_task.json, run_*/multi_run_results.csv, test_metrics.json, top_run_per_model_family_by_{RANK_BY}.csv |  |
| worm_multitask_results_comparison.ipynb | 11 | 8 | IPython.display, __future__, json, math, matplotlib.pyplot, numpy, pandas, pathlib, re | all_classification_reports_long.csv, class_level_reports_long.csv, classification_report_{task}.csv, config.json, confusion_matrix_{task}.csv, history.csv, label_to_index_by_task.json, run_comparison_summary.csv, split_summary.json, test_metrics.json |  |
| worm_same_individual_predictions_top_models.ipynb | 19 | 2 | IPython.display, PIL, __future__, copy, json, matplotlib.pyplot, numpy, pandas, pathlib, re, shlex, sklearn.metrics, sklearn.model_selection, src.dataset_multitask, src.models, torch, torch.nn, torch.nn.functional, tqdm.auto, warnings | /mnt/extssd/Earthworms/petridish-worm-images/01_Segmented/global_metadata.csv, all_top_model_same_individual_image_predictions.csv, all_top_model_same_individual_summary.csv, all_top_model_test_set_image_predictions_long.csv, best_model.pt, classification_report_{task}.csv, cm_scores_image_vs_individual.csv, config.json, confusion_matrix_{task}.csv, consensus_disagreement_rate_by_task.csv, history.csv, image_level_cm_{task}.csv, image_level_scores_on_sample.csv, individual_level_highest_confidence_cm_{task}.csv, individual_level_scores_on_sample.csv, individual_predictions_highest_confidence.csv, individual_predictions_highest_confidence_all_models.csv, label_to_index_by_task.json, merged_test_predictions_wide_all_tasks.csv, merged_test_predictions_wide_{task}.csv, model_disagreement_summary.csv, pairwise_model_disagreement_{task}_{level}.csv, reference_test_rows_recreated_from_individual_split.csv, run_*/multi_run_results.csv, same_individual_image_predictions.csv, sampled_individuals_and_images.csv, sampling_summary.csv, test_cm_{task}.csv, test_metrics.json, test_set_cm_scores_by_model_and_task.csv, test_set_image_predictions.csv, top_run_per_model_family_by_{RANK_BY}.csv | != wide.loc[valid, m2].astype(str) |
| worm_species_sweep_analysis.ipynb | 21 | 14 | IPython.display, json, matplotlib.pyplot, numpy, pandas, pathlib, re, shlex, textwrap, warnings | all_sweep_results_aggregated.csv, best_model.pt, classification_report_{task}.csv, config.json, confusion_matrix_{task}.csv, history.csv, label_to_index_by_task.json, run_*/multi_run_results.csv, test_metrics.json, top_models_by_{RANK_BY}.csv, top_run_per_model_type_by_{RANK_BY}.csv |  |
| worm_umap_top_model_families.ipynb | 13 | 12 | IPython.display, PIL, __future__, copy, json, matplotlib.pyplot, numpy, pandas, pathlib, re, shlex, sklearn.decomposition, sklearn.metrics, src.dataset_multitask, src.models, torch, torch.nn, torch.nn.functional, tqdm.auto, umap, warnings | /mnt/extssd/Earthworms/petridish-worm-images/01_Segmented/global_metadata.csv, best_model.pt, classification_report_{task}.csv, combined_umap_metadata_with_predictions.csv, config.json, confusion_matrix_{task}.csv, history.csv, label_to_index_by_task.json, run_*/multi_run_results.csv, test_metrics.json, top_run_per_model_family_by_{RANK_BY}.csv, umap_metadata_with_predictions.csv |  |

## Duplicate basenames

| Basename | Locations |
| --- | --- |
| README.md | .codex/codex_repository_refactor_handoff/references/README.md<br>README.md |
| __init__.py | src/worm_species/__init__.py<br>src/worm_species/config/__init__.py<br>src/worm_species/data/__init__.py<br>src/worm_species/experiments/__init__.py<br>src/worm_species/models/__init__.py |
| collect_dual_cue_results.py | .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/collect_dual_cue_results.py<br>collect_dual_cue_results.py |
| dataset_multitask.py | .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/dataset_multitask.py<br>.codex/codex_repository_refactor_handoff/references/known_files/original/dataset_multitask.py<br>src/dataset_multitask.py |
| dual_cue_experiment_plan.json | refactor_contract_after/dual_cue_experiment_plan.json<br>refactor_contract_before/dual_cue_experiment_plan.json |
| generate_dual_cue_run_specs.py | .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/generate_dual_cue_run_specs.py<br>generate_dual_cue_run_specs.py |
| submit_dual_cue_experiment.sh | .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/submit_dual_cue_experiment.sh<br>submit_dual_cue_experiment.sh |
| sweep_plan.tsv | refactor_contract_after/sweep_plan.tsv<br>refactor_contract_before/sweep_plan.tsv |
| train_multitask_cue_suppression.py | .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/train_multitask_cue_suppression.py<br>train_multitask_cue_suppression.py |
| train_multitask_masked_hloss.py | .codex/codex_repository_refactor_handoff/references/known_files/original/train_multitask_masked_hloss.py<br>train_multitask_masked_hloss.py |

## Python modules

| Path | Functions | Classes | Imports | From imports | CLI flags |
| --- | --- | --- | --- | --- | --- |
| .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/collect_dual_cue_results.py | read_json, collect_nested_csv, matched_results_long, add_equivalent_condition_aliases, build_comparison, main |  | argparse, json, math, pandas | __future__:annotations, pathlib:Path |  |
| .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/dataset_multitask.py | resolve_path, is_missing_label, strip_final_number, parse_taxonomy_from_barcode, apply_taxonomic_uncertainty_rules, mask_rare_classes_per_task, _missing_values_from_cfg, _normalise_missing_series, _clean_label_value, get_target_cols, _derive_taxonomy_and_stage, prepare_metadata, foreground_bbox_from_image, foreground_bbox_from_mask, pad_square_bbox, build_condition_transform, build_test_condition_transform, build_transforms, is_valid_image, is_valid_image, __init__, __call__, __repr__, __init__, __call__, __repr__, __init__, __call__, __repr__, __init__ | ColourRetention, ChannelShuffle, TensorGaussianBlur, TensorBilateralFilter, PatchShuffle, MultiTaskWormImageDataset | cv2, math, numpy, pandas, re, torch | PIL:Image, __future__:annotations, pathlib:Path, torch.utils.data:Dataset, torchvision.transforms:functional, torchvision:transforms, typing:Any |  |
| .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/generate_dual_cue_run_specs.py | inclusive_sequence, slug, format_override, generate_conditions, sweep_combinations, condition_overrides, main |  | argparse, itertools, json, math, re, yaml | __future__:annotations, pathlib:Path, typing:Any |  |
| .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/train_multitask_cue_suppression.py | parse_sweep_item, generate_colour_retention_values, get_sweep_parameters_from_config, get_sweep_parameters_from_cli, generate_sweep_configs, build_multitask_model, build_label_maps, read_csvs_from_dir, make_loaders, compute_individual_class_weights, build_criteria, _safe_metric, infer_parent_label_from_child_label, build_child_to_parent_matrix, hierarchy_consistency_loss, run_epoch, _inclusive_float_sequence, generate_test_cue_conditions, _test_condition_signature, make_test_condition_loader, evaluate_test_cue_suppression, _wandb_metrics, _flatten_wandb_config, initialise_wandb_run, _score_for_selection, get_input_condition, get_colour_metadata, make_experiment_run_name, train_one_run, main | MultiTaskClassifier | argparse, copy, itertools, json, math, numpy, os, pandas, torch, torch.nn.functional, wandb | __future__:annotations, pathlib:Path, sklearn.metrics:accuracy_score,balanced_accuracy_score,f1_score,classification_report,confusion_matrix, src.cache:build_image_cache, src.dataset_multitask:MISSING_LABEL,MultiTaskWormImageDataset,build_transforms,build_condition_transform,build_test_condition_transform,get_target_cols,prepare_metadata, src.models:build_model, src.splits:make_individual_level_splits, src.utils:load_config,apply_overrides,set_nested,parse_scalar,set_seed,save_json,make_run_name, torch.utils.data:DataLoader, torch:nn | --config, --override, --sweep |
| .codex/codex_repository_refactor_handoff/references/known_files/original/dataset_multitask.py | resolve_path, is_missing_label, strip_final_number, parse_taxonomy_from_barcode, apply_taxonomic_uncertainty_rules, mask_rare_classes_per_task, _missing_values_from_cfg, _normalise_missing_series, _clean_label_value, get_target_cols, _derive_taxonomy_and_stage, prepare_metadata, foreground_bbox_from_image, foreground_bbox_from_mask, pad_square_bbox, build_transforms, is_valid_image, is_valid_image, __init__, __len__, _encode_label, __getitem__ | MultiTaskWormImageDataset | numpy, pandas, re, torch | PIL:Image, __future__:annotations, pathlib:Path, torch.utils.data:Dataset, torchvision:transforms, typing:Any |  |
| .codex/codex_repository_refactor_handoff/references/known_files/original/train_multitask_masked_hloss.py | parse_sweep_item, get_sweep_parameters_from_config, get_sweep_parameters_from_cli, generate_sweep_configs, build_multitask_model, build_label_maps, read_csvs_from_dir, make_loaders, compute_individual_class_weights, build_criteria, _safe_metric, infer_parent_label_from_child_label, build_child_to_parent_matrix, hierarchy_consistency_loss, run_epoch, _wandb_metrics, _flatten_wandb_config, initialise_wandb_run, _score_for_selection, train_one_run, main, __init__, _remove_classifier_and_get_feature_dim, forward | MultiTaskClassifier | argparse, copy, itertools, math, numpy, os, pandas, torch, torch.nn.functional, wandb | __future__:annotations, pathlib:Path, sklearn.metrics:accuracy_score,balanced_accuracy_score,f1_score,classification_report,confusion_matrix, src.cache:build_image_cache, src.dataset_multitask:MISSING_LABEL,MultiTaskWormImageDataset,build_transforms,get_target_cols,prepare_metadata, src.models:build_model, src.splits:make_individual_level_splits, src.utils:load_config,apply_overrides,set_nested,parse_scalar,set_seed,save_json,make_run_name, torch.utils.data:DataLoader, torch:nn | --config, --override, --sweep |
| .codex/codex_repository_refactor_handoff/tools/audit_repository.py | sha256_file, is_excluded_directory_name, excluded_component, is_excluded_directory, git_paths, iter_untracked_source_files, find_excluded_directories, read_text, dotted_name, literal_strings, analyse_python, analyse_shell, extract_notebook_code, analyse_notebook, category_for, make_record, markdown_table, build_markdown, parse_args, main, cell | FileRecord, ExcludedDirectory | argparse, ast, hashlib, json, os, re, subprocess | __future__:annotations, collections:Counter,defaultdict, dataclasses:asdict,dataclass,field, pathlib:Path, typing:Any,Iterable | --ignore, --output-dir |
| .codex/codex_repository_refactor_handoff/tools/compare_contract_trees.py | files_under, json_key_tree, csv_columns, checkpoint_keys, compare_file, parse_args, main, add | Difference, Report | argparse, csv, json, torch | __future__:annotations, dataclasses:dataclass,field, pathlib:Path, typing:Any | --allow-extra |
| collect_dual_cue_results.py |  |  |  | src.worm_species.experiments.result_collection:add_equivalent_condition_aliases,build_comparison,collect_nested_csv,collect_results,main,matched_results_long,read_json |  |
| generate_dual_cue_run_specs.py |  |  |  | src.worm_species.experiments.conditions:condition_overrides,format_override,generate_conditions,inclusive_sequence,slug,sweep_combinations, src.worm_species.experiments.run_specs:main,write_run_specs |  |
| src/cache.py | _file_stamp, _make_cache_key, _cache_one_image, build_image_cache |  | hashlib, os, pandas, tempfile | PIL:Image, __future__:annotations, concurrent.futures:ProcessPoolExecutor,as_completed, pathlib:Path, src.dataset:resolve_path,foreground_bbox_from_image,foreground_bbox_from_mask,pad_square_bbox, tqdm.auto:tqdm, typing:Any |  |
| src/dataset.py | resolve_path, prepare_metadata, foreground_bbox_from_image, foreground_bbox_from_mask, pad_square_bbox, build_transforms, is_valid_image, __init__, __len__, __getitem__ | WormImageDataset | numpy, pandas, torch | PIL:Image, __future__:annotations, pathlib:Path, torch.utils.data:Dataset, torchvision:transforms |  |
| src/dataset_multitask.py | resolve_path, is_missing_label, strip_final_number, parse_taxonomy_from_barcode, apply_taxonomic_uncertainty_rules, mask_rare_classes_per_task, _missing_values_from_cfg, _normalise_missing_series, _clean_label_value, get_target_cols, _derive_taxonomy_and_stage, prepare_metadata, foreground_bbox_from_image, foreground_bbox_from_mask, pad_square_bbox, build_condition_transform, build_test_condition_transform, build_transforms, is_valid_image, is_valid_image, __init__, __call__, __repr__, __init__, __call__, __repr__, __init__, __call__, __repr__, __init__ | ColourRetention, ChannelShuffle, TensorGaussianBlur, TensorBilateralFilter, PatchShuffle, MultiTaskWormImageDataset | cv2, math, numpy, pandas, re, torch | PIL:Image, __future__:annotations, pathlib:Path, torch.utils.data:Dataset, torchvision.transforms:functional, torchvision:transforms, typing:Any |  |
| src/download_pretrained_from_config.py | set_nested, generate_sweep_configs, collect_pretrained_model_names, download_torchvision_model, main |  | argparse, copy, itertools, os, torchvision.models, yaml | __future__:annotations, pathlib:Path, typing:Any | --config, --torch-home |
| src/generate_sweep_run_specs.py | format_value, main |  | argparse, itertools, yaml | __future__:annotations, pathlib:Path, typing:Any | --config, --out-dir |
| src/models.py |  |  |  | src.worm_species.models.factory:_load_model,build_model |  |
| src/splits.py | make_individual_level_splits |  | pandas | __future__:annotations, pathlib:Path, sklearn.model_selection:StratifiedShuffleSplit |  |
| src/utils.py | set_seed, save_json, short_hash, make_run_name |  | hashlib, json, numpy, random, torch | __future__:annotations, pathlib:Path, src.worm_species.config.loading:load_config, src.worm_species.config.overrides:apply_overrides,parse_scalar,set_nested, typing:Any |  |
| src/worm_species/__init__.py |  |  |  |  |  |
| src/worm_species/config/__init__.py |  |  |  | loading:load_config, overrides:apply_overrides,parse_scalar,set_nested, sweeps:generate_colour_retention_values,generate_sweep_configs,get_colour_sweep_parameters_from_config,get_sweep_parameters_from_cli,get_sweep_parameters_from_config,parse_sweep_item |  |
| src/worm_species/config/loading.py | load_config |  | yaml | __future__:annotations, pathlib:Path, typing:Any |  |
| src/worm_species/config/overrides.py | parse_scalar, set_nested, apply_overrides |  |  | __future__:annotations, copy:deepcopy, typing:Any |  |
| src/worm_species/config/sweeps.py | parse_sweep_item, get_sweep_parameters_from_config, generate_colour_retention_values, get_colour_sweep_parameters_from_config, get_sweep_parameters_from_cli, generate_sweep_configs |  | copy, itertools | __future__:annotations, overrides:parse_scalar,set_nested, typing:Any |  |
| src/worm_species/data/__init__.py |  |  |  | labels:build_label_maps,read_csvs_from_dir |  |
| src/worm_species/data/labels.py | build_label_maps, read_csvs_from_dir |  | os, pandas | __future__:annotations, pathlib:Path |  |
| src/worm_species/experiments/__init__.py |  |  |  |  |  |
| src/worm_species/experiments/conditions.py | inclusive_sequence, slug, format_override, generate_conditions, sweep_combinations, condition_overrides |  | itertools, math, re | __future__:annotations, typing:Any |  |
| src/worm_species/experiments/result_collection.py | read_json, collect_nested_csv, matched_results_long, add_equivalent_condition_aliases, build_comparison, collect_results, main |  | argparse, json, pandas | __future__:annotations, pathlib:Path |  |
| src/worm_species/experiments/run_specs.py | write_run_specs, main |  | argparse, json, yaml | __future__:annotations, conditions:condition_overrides,format_override,generate_conditions,sweep_combinations, pathlib:Path |  |
| src/worm_species/models/__init__.py |  |  |  | factory:build_model, multitask:MultiTaskClassifier,build_multitask_model |  |
| src/worm_species/models/factory.py | _load_model, build_model |  | torch.nn | __future__:annotations, torchvision:models |  |
| src/worm_species/models/multitask.py | build_multitask_model, __init__, _remove_classifier_and_get_feature_dim, forward | MultiTaskClassifier | torch | __future__:annotations, factory:build_model, torch:nn |  |
| tests/test_behaviour_contracts.py | torch_save_dict_keys, __init__, __len__, __getitem__, __init__, forward, test_scalar_and_override_contract, test_sweep_product_and_run_name_contract, minimal_config, test_condition_endpoint_deduplication, test_one_and_two_model_run_specs_disable_nested_expansion, test_predefined_split_path_contract, test_transform_order_and_deterministic_test_transform, test_missing_label_and_class_map_contract, test_multitask_head_names_and_shapes, test_hierarchy_mapping_and_zero_consistency_loss, test_metric_keys_and_macro_f1, test_checkpoint_top_level_schemas, test_matched_stress_join_contract, test_legacy_cli_help_flags, test_shell_syntax, test_notebook_parse_and_code_cell_contract, __init__, forward | FixedDataset, FixedModel, ConfigAndSweepContracts, ConditionContracts, TransformAndLabelContracts, LossMetricAndCheckpointContracts, CollectionAndInterfaceContracts, Backbone | ast, collect_dual_cue_results, generate_dual_cue_run_specs, json, math, numpy, pandas, subprocess, sys, tempfile, torch, train_multitask_masked_hloss, unittest, yaml | PIL:Image, __future__:annotations, pathlib:Path, src.dataset_multitask:MISSING_LABEL,ColourRetention,build_condition_transform,build_test_condition_transform,is_missing_label, src.utils:apply_overrides,make_run_name,parse_scalar,short_hash, src.worm_species.data.labels:read_csvs_from_dir, src.worm_species.models.multitask:MultiTaskClassifier, torch.utils.data:DataLoader,Dataset, torch:nn |  |
| train_multitask_colour_ablation.py | generate_sweep_configs, make_loaders, compute_individual_class_weights, build_criteria, _safe_metric, infer_parent_label_from_child_label, build_child_to_parent_matrix, hierarchy_consistency_loss, run_epoch, _wandb_metrics, _flatten_wandb_config, initialise_wandb_run, _score_for_selection, get_colour_metadata, make_experiment_run_name, create_wandb_confusion_matrix, train_one_run, main |  | argparse, copy, itertools, math, numpy, os, pandas, torch, torch.nn.functional, wandb | __future__:annotations, pathlib:Path, sklearn.metrics:accuracy_score,balanced_accuracy_score,f1_score,classification_report,confusion_matrix, src.cache:build_image_cache, src.dataset_multitask:MISSING_LABEL,MultiTaskWormImageDataset,build_transforms,get_target_cols,prepare_metadata, src.splits:make_individual_level_splits, src.utils:load_config,apply_overrides,set_nested,parse_scalar,set_seed,save_json,make_run_name, src.worm_species.config.sweeps:generate_colour_retention_values,generate_sweep_configs,get_colour_sweep_parameters_from_config,get_sweep_parameters_from_cli,parse_sweep_item, src.worm_species.data.labels:build_label_maps,read_csvs_from_dir, src.worm_species.models.multitask:MultiTaskClassifier,build_multitask_model, torch.utils.data:DataLoader, torch:nn | --config, --override, --sweep |
| train_multitask_cue_suppression.py | generate_sweep_configs, make_loaders, compute_individual_class_weights, build_criteria, _safe_metric, infer_parent_label_from_child_label, build_child_to_parent_matrix, hierarchy_consistency_loss, run_epoch, _inclusive_float_sequence, generate_test_cue_conditions, _test_condition_signature, make_test_condition_loader, evaluate_test_cue_suppression, _wandb_metrics, _flatten_wandb_config, initialise_wandb_run, _score_for_selection, get_input_condition, get_colour_metadata, make_experiment_run_name, train_one_run, main, record_condition |  | argparse, copy, itertools, json, math, numpy, os, pandas, torch, torch.nn.functional, wandb | __future__:annotations, pathlib:Path, sklearn.metrics:accuracy_score,balanced_accuracy_score,f1_score,classification_report,confusion_matrix, src.cache:build_image_cache, src.dataset_multitask:MISSING_LABEL,MultiTaskWormImageDataset,build_transforms,build_condition_transform,build_test_condition_transform,get_target_cols,prepare_metadata, src.splits:make_individual_level_splits, src.utils:load_config,apply_overrides,set_nested,parse_scalar,set_seed,save_json,make_run_name, src.worm_species.config.sweeps:generate_colour_retention_values,generate_sweep_configs,get_colour_sweep_parameters_from_config,get_sweep_parameters_from_cli,parse_sweep_item, src.worm_species.data.labels:build_label_maps,read_csvs_from_dir, src.worm_species.models.multitask:MultiTaskClassifier,build_multitask_model, torch.utils.data:DataLoader, torch:nn | --config, --override, --sweep |
| train_multitask_masked.py | make_loaders, compute_individual_class_weights, build_criteria, _safe_metric, run_epoch, _score_for_selection, train_one_run, main |  | argparse, copy, itertools, math, numpy, os, pandas, torch | __future__:annotations, pathlib:Path, sklearn.metrics:accuracy_score,balanced_accuracy_score,f1_score,classification_report,confusion_matrix, src.cache:build_image_cache, src.dataset_multitask:MISSING_LABEL,MultiTaskWormImageDataset,build_transforms,get_target_cols,prepare_metadata, src.splits:make_individual_level_splits, src.utils:load_config,apply_overrides,set_nested,parse_scalar,set_seed,save_json,make_run_name, src.worm_species.config.sweeps:generate_sweep_configs,get_sweep_parameters_from_cli,get_sweep_parameters_from_config,parse_sweep_item, src.worm_species.data.labels:build_label_maps,read_csvs_from_dir, src.worm_species.models.multitask:MultiTaskClassifier,build_multitask_model, torch.utils.data:DataLoader, torch:nn | --config, --override, --sweep |
| train_multitask_masked_hloss.py | make_loaders, compute_individual_class_weights, build_criteria, _safe_metric, infer_parent_label_from_child_label, build_child_to_parent_matrix, hierarchy_consistency_loss, run_epoch, _score_for_selection, train_one_run, main |  | argparse, copy, itertools, math, numpy, os, pandas, torch, torch.nn.functional | __future__:annotations, pathlib:Path, sklearn.metrics:accuracy_score,balanced_accuracy_score,f1_score,classification_report,confusion_matrix, src.cache:build_image_cache, src.dataset_multitask:MISSING_LABEL,MultiTaskWormImageDataset,build_transforms,get_target_cols,prepare_metadata, src.splits:make_individual_level_splits, src.utils:load_config,apply_overrides,set_nested,parse_scalar,set_seed,save_json,make_run_name, src.worm_species.config.sweeps:generate_sweep_configs,get_sweep_parameters_from_cli,get_sweep_parameters_from_config,parse_sweep_item, src.worm_species.data.labels:build_label_maps,read_csvs_from_dir, src.worm_species.models.multitask:MultiTaskClassifier,build_multitask_model, torch.utils.data:DataLoader, torch:nn | --config, --override, --sweep |
| train_multitask_masked_hloss_wandb.py | make_loaders, compute_individual_class_weights, build_criteria, _safe_metric, infer_parent_label_from_child_label, build_child_to_parent_matrix, hierarchy_consistency_loss, run_epoch, _wandb_metrics, _flatten_wandb_config, initialise_wandb_run, _score_for_selection, train_one_run, main |  | argparse, copy, itertools, math, numpy, os, pandas, torch, torch.nn.functional, wandb | __future__:annotations, pathlib:Path, sklearn.metrics:accuracy_score,balanced_accuracy_score,f1_score,classification_report,confusion_matrix, src.cache:build_image_cache, src.dataset_multitask:MISSING_LABEL,MultiTaskWormImageDataset,build_transforms,get_target_cols,prepare_metadata, src.splits:make_individual_level_splits, src.utils:load_config,apply_overrides,set_nested,parse_scalar,set_seed,save_json,make_run_name, src.worm_species.config.sweeps:generate_sweep_configs,get_sweep_parameters_from_cli,get_sweep_parameters_from_config,parse_sweep_item, src.worm_species.data.labels:build_label_maps,read_csvs_from_dir, src.worm_species.models.multitask:MultiTaskClassifier,build_multitask_model, torch.utils.data:DataLoader, torch:nn | --config, --override, --sweep |

## Parse/read warnings

| Path | Warning |
| --- | --- |
| cam_mutlitask.ipynb | JSONDecodeError: Expecting value: line 1 column 1 (char 0) |

## All files

| Path | Tracked | Category | Lines | Bytes | Executable | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| .codex/codex_repository_refactor_handoff/AGENTS.md | False | documentation | 148 | 6754 | False | f79e2f4ee15eab0f4317d529ef5b3a80e8e26598e057dd3cd46dba813910d09a |
| .codex/codex_repository_refactor_handoff/CODEX_PROMPT.txt | False | documentation | 1 | 819 | False | 1d8e09f05197d3d04a9a80a2b4abed4ab084fac3723be216005db35c2f5d4b48 |
| .codex/codex_repository_refactor_handoff/CODEX_TASK.md | False | documentation | 146 | 6220 | False | 7af6db94c24da924e8b10cdd9e0a7d845954f56e481e09521c7aaaed5346c780 |
| .codex/codex_repository_refactor_handoff/START_HERE.md | False | documentation | 23 | 1433 | False | 423e7f88e0f9a2f84aa938d2f580cf355f46c03a9077abaf8c696dadf094fdd4 |
| .codex/codex_repository_refactor_handoff/docs/BEHAVIOUR_CONTRACT.md | False | documentation | 91 | 3152 | False | 972e6a083860376ea1ee60f6b14ea6263069afde4753b8e707c545005c3ffb17 |
| .codex/codex_repository_refactor_handoff/docs/REPOSITORY_WIDE_INVENTORY_REQUIREMENTS.md | False | documentation | 74 | 2437 | False | eec516f490ee88a6395bf4d91981d5b395462d885e84513b8c7dddbd6a558b59 |
| .codex/codex_repository_refactor_handoff/docs/TARGET_ARCHITECTURE.md | False | documentation | 70 | 2616 | False | 9e1811745c273ffd4c9e3a25b8b00cb3a40abe01330c6cc3ed9713dcd2c13fa7 |
| .codex/codex_repository_refactor_handoff/docs/VALIDATION_CHECKLIST.md | False | documentation | 69 | 2560 | False | 5652f692d37842674b007763f3bad5d661f4f037a6496570b34dd1f6158e375b |
| .codex/codex_repository_refactor_handoff/references/KNOWN_FILES_MANIFEST.tsv | False | other | None | 1407 | False | de180a45a5a45b3a455cf02c4aaac171c98a2c47fddd29d1d0372e13fe627c8e |
| .codex/codex_repository_refactor_handoff/references/README.md | False | documentation | 19 | 910 | False | d958da0d7783becf28f67480fe9e8596297c9d4be846b472f0348886a4717a69 |
| .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/README_dual_cue_experiment.md | False | documentation | 120 | 5313 | False | b31ba41ada78f6a6fd6eb36c32d6210c5918920ee7c995a38b9a9253e1a87575 |
| .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/collect_dual_cue_results.py | False | python | 225 | 7823 | False | 4b11e82686786139f9aad4425086a17de6fb39e1c0b3ce8ebdcf50f9667fc431 |
| .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/dataset_multitask.py | False | python | 1121 | 37786 | False | ff4e37f32af384f2227c8388f1784cb79c8184bef7903485dc2f2ad132162097 |
| .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/dual_cue_experiment_config_snippet.yaml | False | config | 80 | 2019 | False | 52a12b23d2f8ad4c9975e97ba760993dc604744e940d4b0f3dcb22258d5b074d |
| .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/generate_dual_cue_run_specs.py | False | python | 330 | 11675 | False | be036ddfb1e63791e9aee7db681b10e8e7ff7e900606329b3e8a27ced4da6fd1 |
| .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/submit_dual_cue_experiment.sh | False | shell_or_slurm | 737 | 23653 | False | dda1a3c3c51448a742472f44a96156a4ae472c50d609361f289cb596190a735f |
| .codex/codex_repository_refactor_handoff/references/known_files/dual_cue/train_multitask_cue_suppression.py | False | python | 1829 | 65867 | False | 6a2a09dd895dfbc3bc5016e3631783e054f7969a60c2a8f820c05a7d5b1a88e6 |
| .codex/codex_repository_refactor_handoff/references/known_files/original/dataset_multitask.py | False | python | 840 | 27809 | False | 40898d12d86dcfbc5ecf1ef45338c5cbdfac307b484de8fa088ce4e94a88a247 |
| .codex/codex_repository_refactor_handoff/references/known_files/original/submit_node_local_sweep.sh | False | shell_or_slurm | 731 | 22709 | False | 66dc25cbc3baa93f4b2225be0272fbc13129c45a1f3331bab46174b98e861249 |
| .codex/codex_repository_refactor_handoff/references/known_files/original/train_multitask_masked_hloss.py | False | python | 1218 | 42401 | False | 0750ce9a73ac842342c35887fd1019867200ae750f724137fb9319f08b7a970d |
| .codex/codex_repository_refactor_handoff/tools/audit_repository.py | False | python | 660 | 22205 | True | af625924cff2a5839dc72f3eb0b7737a856e6fc6d7a1533ddb8c5b3be91ca3a2 |
| .codex/codex_repository_refactor_handoff/tools/compare_contract_trees.py | False | python | 143 | 4663 | True | ca3c78f3c400bb86488e16a5099de111d90dc9e4dba881c19e99aaad75e8c7f0 |
| .gitignore | True | other | None | 5063 | False | 86bc6789c518b6b8ea4ed9f15eb130477461d369221e478459e12c5cd089c05e |
| 01_build_persistent_cache_resolved.sh | True | shell_or_slurm | 203 | 5790 | True | d4bd31dca0490458d273358de4719eaed3260589e4b725be620c555ae0d152ec |
| 02_submit_sweep_cache_to_tmp_resolved.sh | True | shell_or_slurm | 441 | 13324 | True | 6f1e551a47f3d106e869550386803a652ff7a910a0c4c319306058a60945c67e |
| README.md | True | documentation | 102 | 3603 | False | 00b0e37c63e10de11693469e12c70f66f76d57349573077dbda90069a3839e66 |
| archive_project_to_drive.sh | True | shell_or_slurm | 135 | 4099 | True | 8cf5c8c74248db6f04c553b1addd13b0eacc90bae374587dc1894880bcb6e130 |
| cam_mutlitask.ipynb | True | notebook | 0 | 0 | False | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| collect_dual_cue_results.py | True | python | 25 | 505 | False | 35768e35fa5cfd40e71ac152738e5270f1b21ed87809653ef74a81f5a7a05612 |
| config.yaml | True | config | 207 | 3718 | False | 68b6529aed0e30cf10738011f78ad076d23f1a031bb4dbe461ea09b3f8219868 |
| config_old.yaml | True | config | 147 | 2806 | False | eaaeca953875e1fa4701b7026511c2aef84f75c28161cdf2f42e0febae9aaf5c |
| data_leakage_check.ipynb | True | notebook | 16091 | 2653389 | False | 8b36a2d477fa81a72dd8d62aa76308fdc55862c6618c641c94e000bdcfb322f2 |
| data_notebooks/data_analysis.ipynb | True | notebook | 1383 | 200924 | False | c4536f551092e6b950e94f23c67fb62474458271c31653a409c658ee491782b4 |
| data_notebooks/data_explore.ipynb | True | notebook | 1919 | 4144657 | False | d7d909bb51d8be2f212c58d74ad1aad353b3f9b22b133f01b3970e9f5f98f5c4 |
| dataset_tes.ipynb | True | notebook | 1132 | 374458 | False | 6abef93b7324255f3546ae5242749ffce3aebfc90080b160f2665682b1db60e4 |
| earthworm_colour_robustness_analysis.ipynb | True | notebook | 3756 | 1631331 | False | ec2f8a51224b7ab5aa0282042da3d17b895ee6b79ca3817c1a39c0801edcc4e3 |
| earthworm_cue_suppression_analysis_v2.ipynb | True | notebook | 9948 | 5195781 | False | e61b6385eff8ee33b7d16e310d2bf5d02d19677b3d2e1175d3ff5092b893cfee |
| environment.yaml | True | config | 18 | 276 | False | 194e9b78af293d9ea5a482c1a5980329ce328907c3652c0444b5dbb920e0170c |
| generate_dual_cue_run_specs.py | True | python | 26 | 566 | False | 2d38a2899cb034af5761ae7b14440f99f57964724dad31150bfc6d1c17fcbe8f |
| gradcam_multitask_all_tasks.ipynb | True | notebook | 993 | 35864 | False | 7c61a94aab66de997eb9ae8e55f72383f5ed6aec51967bbeeb27c64e62b0394e |
| refactor_contract_after/dual_cue_experiment_plan.json | False | config | 810 | 18456 | False | 6b71b91a3f9327511d53daf4b5fa3532a5191cdd4b86c5d76e0c7ee5d78aceba |
| refactor_contract_after/sweep_plan.tsv | False | other | None | 79501 | False | 8340964b84ae0677304324384a3e81703e3f5b9829497ee08ecc7849f920954c |
| refactor_contract_before/dual_cue_experiment_plan.json | False | config | 810 | 18456 | False | 6b71b91a3f9327511d53daf4b5fa3532a5191cdd4b86c5d76e0c7ee5d78aceba |
| refactor_contract_before/sweep_plan.tsv | False | other | None | 79501 | False | 8340964b84ae0677304324384a3e81703e3f5b9829497ee08ecc7849f920954c |
| run_persistent_cache_sweep_wandb.sh | True | shell_or_slurm | 480 | 14921 | False | 6ac4cc85b90c3205650723325641036bce15e51bb34e82a09c54bf817986c729 |
| single_task/analyze_species_outputs_v2.ipynb | True | notebook | 12130 | 2112588 | False | 5a4fd3d5911b89c1fc92c38ad0941e0f5f7edc36ac24ae3daaae7bb87c16dde4 |
| single_task/cam.ipynb | True | notebook | 898 | 2353750 | False | c773ee56a412620130e33a222e0c888874369c5b4f0d675abfccba2073c6ffa9 |
| single_task/umap.ipynb | True | notebook | 10 | 119 | False | 9309e50d839e2ef0462015687d5d4b549e0d574d60fc36b5f7df1d7ebdaf36c1 |
| single_task/umap_species_embeddings.ipynb | True | notebook | 220712 | 22929036 | False | 011e70f229b96de77e9e85f9f50ce2c2e3193d600bf34d93e5f52596b5a1c16d |
| split_csv/check_splits copy.ipynb | True | notebook | 150 | 5196 | False | 022f76ee69f18d537edea573c4468f4f086b1c53884f0011656368a23775b60d |
| split_csv/test_split.csv | True | other | None | 860974 | False | 832cd598fd835c9203959d4c508eb2d72cbd1236a65f10b6a4d73018a2da843f |
| split_csv/train_split.csv | True | other | None | 2703362 | False | 001d69ffb2216745bcc5c94609c6d3c5c3f2e629777c3c5763d101990599c84f |
| split_csv/val_split.csv | True | other | None | 632896 | False | 12156d4a83643d5edda79c4978916713b95fb3b7df2628d407bc20d406d87226 |
| src/cache.py | True | python | 201 | 5821 | False | e623eb273822fd21a779f20299dee8bb0240ee2dcc06c379d148e97899de03d6 |
| src/dataset.py | True | python | 237 | 6756 | False | 40d86da9937d29b63c23c48f742e41ed6fe45d8b188070f6f1372cbbbd43144b |
| src/dataset_multitask.py | True | python | 1121 | 37786 | False | ff4e37f32af384f2227c8388f1784cb79c8184bef7903485dc2f2ad132162097 |
| src/download_pretrained_from_config.py | True | python | 203 | 6032 | False | 81ccfcb9c127e878ab1b01a2d93b6a87711cba50dd9f48eab2d1061c18ea1eb7 |
| src/generate_sweep_run_specs.py | True | python | 112 | 2732 | True | 9ad6899591897f561d78cddaea10e1ad494671114e23f0687dbd43e04c2f7b1d |
| src/models.py | True | python | 5 | 187 | False | a68cc77a6f0699999de490a1f581140cc7deda82938821862ed139884373f786 |
| src/splits.py | True | python | 102 | 3290 | False | 9f6194c6c890b1fbf3b2eaa53461a104c0f78664d97ad5e0f51e73d64c2b6472 |
| src/utils.py | True | python | 45 | 1257 | False | 5f4e54079e47a30fa74d6b937ba7833e0584bd870bc87888eace7b72c370a0aa |
| src/worm_species/__init__.py | False | python | 2 | 81 | False | c806fbcb6afd70267b0f2e28d168410c68fa8abf02a61a3954175e9dee160600 |
| src/worm_species/config/__init__.py | False | python | 26 | 711 | False | 9b7f8c88ba70a9330600092e259d2c953f4a4a5eea9fdaa62e4bf4437b0da3f2 |
| src/worm_species/config/loading.py | False | python | 20 | 529 | False | c6e219b13a11eba714c17a724e895b68aec0a966f9716b2a29c2c86666994833 |
| src/worm_species/config/overrides.py | False | python | 55 | 1448 | False | c85a02b719a56da57b01418ed043a557cf9ad64a60ea2311fd9ca7f277b42fc9 |
| src/worm_species/config/sweeps.py | False | python | 121 | 4681 | False | 479ba00a84b96942ceeca9d9f0e704000bfae25dfe24bd0281cf183b57a30031 |
| src/worm_species/data/__init__.py | False | python | 6 | 184 | False | 51e8868891519db10ac6e50493e2451989a5ab023c587f01bf9c6784f21dd779 |
| src/worm_species/data/labels.py | False | python | 54 | 2193 | False | 6ff539ec1f69618ee26c33bc50354d05deb9b44da40686357f36ac40ae19360a |
| src/worm_species/experiments/__init__.py | False | python | 2 | 76 | False | 88074609ce4ff5cc6eb251df946850fc69eeb5e45433883baa7cf47014cfc831 |
| src/worm_species/experiments/conditions.py | False | python | 220 | 8685 | False | 32b618f194ca833cea250e45530bdf2727e007b6dd9ac6a461a0132f90560d63 |
| src/worm_species/experiments/result_collection.py | False | python | 196 | 7656 | False | 54adefa337eaacf7fca649bfa75fb5ae12eba1bdb2668fce4bebba4e04d7b6ad |
| src/worm_species/experiments/run_specs.py | False | python | 74 | 2745 | False | 4c9386eea370bbe15c30aab518237204f9f59399cbfa41ff04a4fcfe0e84b946 |
| src/worm_species/models/__init__.py | False | python | 7 | 245 | False | ea84a4fafc9fca128c236fe1221bfcea1c634fb861633e9217442d7642c603d7 |
| src/worm_species/models/factory.py | False | python | 60 | 2017 | False | 2f2328f9e4529c33f7ddf9210b352299b157741a9f484653b2168dca2b9d157a |
| src/worm_species/models/multitask.py | False | python | 78 | 2999 | False | 1aea371e94b13a65d82b0d44665dcaf1b02c33ace87e37146ac0931eff7dc270 |
| submit_colour_ablation_sweep.sh | True | shell_or_slurm | 876 | 27532 | False | 808e0823340182f34970739f8ae0a2fdf8c8e0b122d212cf58afd8e7ff9289af |
| submit_dual_cue_experiment.sh | True | shell_or_slurm | 737 | 23653 | False | dda1a3c3c51448a742472f44a96156a4ae472c50d609361f289cb596190a735f |
| submit_dual_cue_experiment_genome.sh | True | shell_or_slurm | 646 | 19893 | False | 7d626bb136b381370a8702f15c8ee1b7d1713454781d2a2ba3fde27a33ac89f3 |
| submit_worm_node_local_scratch_sweep.sh | True | shell_or_slurm | 703 | 21582 | True | 1083f1bc033af02246a473caee98bdb09a772626a4ecbdd03ba91e2360a7daf0 |
| submit_worm_node_local_scratch_sweep_hloss.sh | True | shell_or_slurm | 702 | 21599 | True | 545d6a250517d67c6eb0b70bffeea28680f43d40f6ee46fb680ede99f59f7170 |
| tests/test_behaviour_contracts.py | False | python_test | 423 | 16870 | False | 11cf1d2c3d8b8dcd93a0e418eaa0789452214b86eb0032a5ffa57266a0ecc599 |
| train_multitask_colour_ablation.py | True | python | 1156 | 39971 | False | c7586b590f7b661c94a9b78d18628a7191594804520c28ffbb6e525c36e3226c |
| train_multitask_cue_suppression.py | True | python | 1609 | 57630 | False | 55bb5fb8119525c3c1eeb18add264e0407fef9ae67b6eeff22930fb40924b4cb |
| train_multitask_masked.py | True | python | 718 | 23929 | False | afe554b09f0ee95a77602d739be80b6a0ed3eb7d699c4cf7b0ca59cc68b65d8a |
| train_multitask_masked_hloss.py | True | python | 927 | 31497 | False | 9f0840dc47a02384fd346626fc56fc6f3e00960b6824e0a29254b46f6680ae5f |
| train_multitask_masked_hloss_wandb.py | True | python | 1033 | 35790 | False | e6bd145a7c25966fc8fcc18d12e296bb6fef5c6439bdc392d4804e80df3e4747 |
| tranfser_from_genome.sh | True | shell_or_slurm | 10 | 364 | True | cdbbbb6ad950d7d6b3ebd9c41eb8fadaa57788e6a06c891e053959ca9af52f20 |
| tranfser_from_ghpc.sh | True | shell_or_slurm | 10 | 369 | True | 36780cd8c8c0a8bf6a99e35ec9cc5c10ddb9e0a08bc0210bbb52146fde526f0c |
| worm_advanced_test_diagnostics_top_models.ipynb | True | notebook | 8030 | 18534286 | False | 5a088cf1a35d087efc53d3ff321126d2b43ea2cf9a8f23630bb863361c84adbf |
| worm_gradcam_top_model_families.ipynb | True | notebook | 2879 | 11168924 | False | 5fb0fc321c528282951092fccc00c63db5964e6aee593e6a3e0e9a18dffd8e28 |
| worm_multitask_results_comparison.ipynb | True | notebook | 5436 | 5241989 | False | 500c1bd507283af8a6c25a954ba7a18cbc169683455ff1a8f0cdf386c6d231c6 |
| worm_same_individual_predictions_top_models.ipynb | True | notebook | 6859 | 23212104 | False | 12facd3aa48bd39aab9ecfb5e85b891a920036e02c40f06e2b3f20d6c79bcb74 |
| worm_species_sweep_analysis.ipynb | True | notebook | 14516 | 3870457 | False | c4220114b36af9aa399dda8fd5e4e8ee9c4eccb4d8a991484b3bd87b733edf26 |
| worm_umap_top_model_families.ipynb | True | notebook | 2670 | 2934048 | False | dcf48ddbde13da61d6a6086f6e35170e3406be52af07304e98650a5dc27951c5 |
