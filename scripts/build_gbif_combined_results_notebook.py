#!/usr/bin/env python3
"""Build one editable notebook for baseline inference and all training phases."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import nbformat as nbf


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build_notebook(config_path: str, output_path: Path) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    notebook["cells"] = [
        markdown(
            "# GBIF–PETI transfer and domain-order results\n\n"
            "This notebook combines baseline inference with the four fixed transfer "
            "strategies (GBIF only, PETI → GBIF, GBIF → PETI, and mixed), two hierarchy "
            "weights, and three seeds. It reads completed artifacts only and never trains "
            "or submits jobs. GBIF metrics are agreement with occurrence metadata."
        ),
        code(f"""from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
import yaml

config_value = Path(os.environ.get('GBIF_TRAINING_CONFIG', {config_path!r}))
config_candidates = [config_value, Path.cwd() / config_value, Path.cwd().parent / config_value]
CONFIG_PATH = next((path.resolve() for path in config_candidates if path.is_file()), None)
if CONFIG_PATH is None:
    raise FileNotFoundError(f'Could not resolve GBIF training config from: {{config_candidates}}')
config = yaml.safe_load(CONFIG_PATH.read_text())
def expand(value):
    if isinstance(value, dict): return {{k: expand(v) for k, v in value.items()}}
    if isinstance(value, list): return [expand(v) for v in value]
    if isinstance(value, str): return os.path.expandvars(os.path.expanduser(value))
    return value
config = expand(config)
OUTPUT_ROOT = Path(config['paths']['output_root'])
REPORT_ROOT = OUTPUT_ROOT / 'combined_report'
FIGURE_ROOT = REPORT_ROOT / 'figures'
REPORT_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
print('Configuration:', CONFIG_PATH)
print('Results:', OUTPUT_ROOT)
"""),
        markdown("## Dataset and provenance"),
        code("""prepared_summary_path = OUTPUT_ROOT / 'prepared' / 'summary.json'
if prepared_summary_path.is_file():
    prepared_summary = json.loads(prepared_summary_path.read_text())
    display(pd.DataFrame(prepared_summary['rows']).T)
    display(pd.Series(prepared_summary['label_counts'], name='classes').to_frame())
else:
    prepared_summary = None
    print('Pending: prepared dataset summary is not available.')
"""),
        markdown(
            "## Baseline inference on curated GBIF\n\n"
            "The merged prediction CSV contains the row-level results; the summary JSON "
            "contains provenance and coverage metadata only. A `known` scope means that "
            "the GBIF label occurs in the checkpoint label map. Agreement below compares "
            "the model prediction with GBIF occurrence metadata and is **not independently "
            "verified accuracy**."
        ),
        code("""inference_path = OUTPUT_ROOT / 'inference' / 'baseline' / 'predictions.csv'
inference_summary_path = inference_path.with_suffix('.summary.json')
if inference_path.is_file() and inference_summary_path.is_file():
    inference = pd.read_csv(inference_path, dtype=str, keep_default_na=False)
    inference_summary = json.loads(inference_summary_path.read_text())
    display(pd.Series(inference_summary, name='value').to_frame())
else:
    inference = pd.DataFrame()
    inference_summary = None
    print('Pending: merged 12-shard baseline inference is not available.')
    print('Expected predictions:', inference_path)
    print('Expected summary:', inference_summary_path)
"""),
        markdown(
            "### Coverage and agreement with GBIF metadata\n\n"
            "Raw agreement is compared with the majority-label baseline. Balanced "
            "accuracy is the mean recall across known classes, so every class contributes "
            "equally; its unadjusted random-chance reference is `1/K` for `K` evaluated "
            "classes."
        ),
        code("""coverage_rows = []
known_class_recall_rows = []
for task, label_column in (('genus', 'genus'), ('species', 'species_label')):
    scope_column = f'checkpoint_{task}_scope'
    prediction_column = f'predicted_{task}'
    confidence_column = f'predicted_{task}_confidence'
    if inference.empty or scope_column not in inference:
        continue
    scope_values = inference[scope_column].astype(str).str.lower()
    known = scope_values.eq('known')
    confidence = (
        pd.to_numeric(inference[confidence_column], errors='coerce')
        if confidence_column in inference else pd.Series(np.nan, index=inference.index)
    )
    evaluation = pd.DataFrame()
    if label_column in inference and prediction_column in inference:
        valid = (
            known
            & inference[label_column].astype(str).ne('')
            & inference[prediction_column].astype(str).ne('')
        )
        evaluation = pd.DataFrame({
            'label': inference.loc[valid, label_column].astype(str),
            'prediction': inference.loc[valid, prediction_column].astype(str),
        })
        evaluation['correct'] = evaluation['label'].eq(evaluation['prediction'])
    class_recalls = []
    if not evaluation.empty:
        for label, values in evaluation.groupby('label', sort=True):
            correct_rows = int(values['correct'].sum())
            recall = float(values['correct'].mean())
            class_recalls.append(recall)
            known_class_recall_rows.append({
                'task': task, 'label': label, 'rows': int(len(values)),
                'correct_rows': correct_rows, 'recall': recall,
            })
        label_counts = evaluation['label'].value_counts()
        majority_label = str(label_counts.index[0])
        majority_accuracy = float(label_counts.iloc[0] / len(evaluation))
        model_accuracy = float(evaluation['correct'].mean())
        balanced_accuracy = float(np.mean(class_recalls))
        balanced_chance = float(1.0 / len(class_recalls))
    else:
        majority_label = None
        majority_accuracy = np.nan
        model_accuracy = np.nan
        balanced_accuracy = np.nan
        balanced_chance = np.nan
    coverage_rows.append({
        'task': task,
        'rows': int(len(inference)),
        'known_rows': int(known.sum()),
        'unknown_rows': int((~known).sum()),
        'known_coverage': float(known.mean()),
        'evaluated_known_rows': int(len(evaluation)),
        'correct_known_rows': int(evaluation['correct'].sum()) if not evaluation.empty else 0,
        'known_label_agreement': model_accuracy,
        'majority_label': majority_label,
        'majority_accuracy': majority_accuracy,
        'accuracy_minus_majority': model_accuracy - majority_accuracy,
        'evaluated_classes': int(len(class_recalls)),
        'balanced_accuracy': balanced_accuracy,
        'balanced_chance_1_over_k': balanced_chance,
        'balanced_accuracy_minus_chance': balanced_accuracy - balanced_chance,
        'mean_confidence_all': float(confidence.mean()),
        'mean_confidence_known': float(confidence.loc[known].mean()),
        'mean_confidence_unknown': float(confidence.loc[~known].mean()),
    })
baseline_inference_metrics = pd.DataFrame(coverage_rows)
baseline_known_class_recall = pd.DataFrame(known_class_recall_rows)
if not baseline_inference_metrics.empty:
    display(
        baseline_inference_metrics.style.format({
            'known_coverage': '{:.1%}',
            'known_label_agreement': '{:.1%}',
            'majority_accuracy': '{:.1%}',
            'accuracy_minus_majority': '{:+.1%}',
            'balanced_accuracy': '{:.1%}',
            'balanced_chance_1_over_k': '{:.1%}',
            'balanced_accuracy_minus_chance': '{:+.1%}',
            'mean_confidence_all': '{:.3f}',
            'mean_confidence_known': '{:.3f}',
            'mean_confidence_unknown': '{:.3f}',
        }, na_rep='not available')
    )
    baseline_inference_metrics.to_csv(
        REPORT_ROOT / 'baseline_inference_metrics.csv', index=False
    )
    if not baseline_known_class_recall.empty:
        display(baseline_known_class_recall.style.format({'recall': '{:.1%}'}))
        baseline_known_class_recall.to_csv(
            REPORT_ROOT / 'baseline_known_class_recall.csv', index=False
        )
    print('Agreement is with GBIF occurrence metadata, not independently verified accuracy.')
else:
    print('Pending: row-level inference columns are not available.')
"""),
        markdown("### Prediction and confidence distributions"),
        code("""prediction_distribution_rows = []
confidence_summary_rows = []
confidence_by_task = {}
for task in ('genus', 'species', 'age'):
    prediction_column = f'predicted_{task}'
    confidence_column = f'predicted_{task}_confidence'
    if inference.empty or prediction_column not in inference:
        continue
    predictions = inference[prediction_column].replace('', pd.NA).dropna()
    counts = predictions.value_counts()
    shown = counts.head(15)
    for rank, (label, count) in enumerate(shown.items(), start=1):
        prediction_distribution_rows.append({
            'task': task, 'rank': rank, 'predicted_label': label,
            'rows': int(count), 'fraction': float(count / len(predictions)),
        })
    other = int(counts.iloc[15:].sum())
    if other:
        prediction_distribution_rows.append({
            'task': task, 'rank': 16, 'predicted_label': 'Other labels',
            'rows': other, 'fraction': float(other / len(predictions)),
        })
    if confidence_column not in inference:
        continue
    confidence = pd.to_numeric(inference[confidence_column], errors='coerce')
    scope_column = f'checkpoint_{task}_scope'
    scope = (
        inference[scope_column].astype(str).str.lower()
        if scope_column in inference else pd.Series('all rows', index=inference.index)
    )
    confidence_by_task[task] = (confidence, scope)
    for scope_name in sorted(scope.dropna().unique()):
        values = confidence.loc[scope.eq(scope_name)].dropna()
        if values.empty:
            continue
        confidence_summary_rows.append({
            'task': task, 'scope': scope_name, 'rows': int(len(values)),
            'mean': float(values.mean()), 'median': float(values.median()),
            'q10': float(values.quantile(0.10)), 'q90': float(values.quantile(0.90)),
        })

prediction_distribution = pd.DataFrame(prediction_distribution_rows)
confidence_summary = pd.DataFrame(confidence_summary_rows)
if not prediction_distribution.empty:
    display(prediction_distribution.style.format({'fraction': '{:.1%}'}))
    prediction_distribution.to_csv(
        REPORT_ROOT / 'baseline_prediction_distribution.csv', index=False
    )
if not confidence_summary.empty:
    display(confidence_summary.style.format({
        'mean': '{:.3f}', 'median': '{:.3f}', 'q10': '{:.3f}', 'q90': '{:.3f}'
    }))
    confidence_summary.to_csv(REPORT_ROOT / 'baseline_confidence_summary.csv', index=False)

if confidence_by_task:
    fig, axes = plt.subplots(
        1, len(confidence_by_task), figsize=(6 * len(confidence_by_task), 4), squeeze=False
    )
    colours = {'known': '#0072B2', 'unknown': '#D55E00', 'all rows': '#009E73'}
    bins = np.linspace(0, 1, 21)
    for ax, (task, (confidence, scope)) in zip(axes[0], confidence_by_task.items()):
        for scope_name in sorted(scope.dropna().unique()):
            values = confidence.loc[scope.eq(scope_name)].dropna().to_numpy()
            if not len(values):
                continue
            ax.hist(
                values, bins=bins, weights=np.ones(len(values)) / len(values),
                histtype='step', linewidth=2, color=colours.get(scope_name, '0.35'),
                label=f'{scope_name} (n={len(values):,})',
            )
        ax.set_xlim(0, 1)
        ax.set_xlabel('Top-1 model confidence')
        ax.set_ylabel('Within-scope fraction per bin')
        ax.set_title(task.capitalize())
        ax.grid(axis='y', alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle('Baseline inference confidence distributions')
    fig.tight_layout()
    for suffix in config['reporting']['formats']:
        fig.savefig(FIGURE_ROOT / f'baseline_inference_confidence.{suffix}', bbox_inches='tight')
    plt.show()
else:
    print('Pending: prediction confidence columns are not available.')
"""),
        markdown("### Representative prediction rows"),
        code("""if not inference.empty:
    representative_columns = [column for column in (
        'image_id', 'gbif_id', 'genus', 'predicted_genus',
        'predicted_genus_confidence', 'checkpoint_genus_scope', 'genus_label_agreement',
        'species_label', 'predicted_species', 'predicted_species_confidence',
        'checkpoint_species_scope', 'species_label_agreement',
        'predicted_age', 'predicted_age_confidence',
    ) if column in inference]
    representative_predictions = (
        inference.sort_values('image_id').sample(
            n=min(20, len(inference)), random_state=2026
        ).sort_values('image_id')[representative_columns]
    )
    display(representative_predictions)
    representative_predictions.to_csv(
        REPORT_ROOT / 'baseline_representative_predictions.csv', index=False
    )
    print('Rows are a deterministic random sample (seed 2026), not hand-selected examples.')
else:
    representative_predictions = pd.DataFrame()
"""),
        markdown("## Collect all completed training stages"),
        code("""rows = []
for metrics_path in sorted((OUTPUT_ROOT / 'runs').glob('*/*/seed-*/*/**/test_metrics.json')):
    run_dir = metrics_path.parent
    status_path = run_dir / 'run_status.json'
    spec_path = run_dir / 'spec.json'
    if not status_path.is_file() or not spec_path.is_file():
        continue
    status = json.loads(status_path.read_text())
    if status.get('status') != 'complete':
        continue
    spec = json.loads(spec_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    for domain in ('gbif', 'petri'):
        for task in ('genus', 'species', 'age'):
            value = metrics.get(domain, {}).get(f'{task}_macro_f1')
            n = metrics.get(domain, {}).get(f'{task}_n')
            if value is None:
                continue
            rows.append({
                **{k: spec.get(k, '') for k in ('phase', 'model', 'seed', 'strategy', 'regime', 'stage', 'domain')},
                'hierarchy_loss_weight': float(spec.get('hierarchy_loss_weight', spec.get('hierarchy_loss', {}).get('weight', 0.0))),
                'evaluation_domain': domain, 'task': task,
                'macro_f1': float(value), 'n': int(n),
                'final_model': bool(spec.get('final_model', False)),
                'stopped_early': bool(status.get('stopped_early', False)),
                'completed_steps': int(status.get('stage_step', 0)),
            })
metrics = pd.DataFrame(rows)
print(f'Collected {len(metrics)} completed domain-task rows.')
display(metrics.head()) if not metrics.empty else print('Pending: no completed training stages.')
if not metrics.empty:
    metrics.to_csv(REPORT_ROOT / 'all_stage_metrics.csv', index=False)
"""),
        markdown("## Final model comparison"),
        code("""final_metrics = metrics.loc[metrics['final_model']].copy() if not metrics.empty else pd.DataFrame()
if not final_metrics.empty:
    summary = final_metrics.groupby(
        ['phase', 'model', 'strategy', 'hierarchy_loss_weight', 'evaluation_domain', 'task'], as_index=False
    ).agg(mean_macro_f1=('macro_f1', 'mean'), sd=('macro_f1', 'std'), seeds=('seed', 'nunique'))
    summary['ci95'] = student_t.ppf(0.975, summary['seeds'] - 1) * summary['sd'] / np.sqrt(summary['seeds'])
    summary.to_csv(REPORT_ROOT / 'final_model_summary.csv', index=False)
    display(summary)

    label_maps_path = OUTPUT_ROOT / 'prepared' / 'label_maps.json'
    label_maps = json.loads(label_maps_path.read_text())
    chance = {}
    for domain in ('gbif', 'petri'):
        test_path = OUTPUT_ROOT / 'prepared' / f'{domain}_test.csv'
        if not test_path.is_file(): continue
        test = pd.read_csv(test_path, dtype=str, keep_default_na=False)
        for task in ('genus', 'species', 'age'):
            observed = test[task].loc[test[task].ne('')]
            if observed.empty: continue
            k = len(label_maps[task])
            prevalence = observed.value_counts(normalize=True)
            q = 1.0 / k
            chance[(domain, task)] = float((2 * prevalence * q / (prevalence + q)).mean())

    panels = list(final_metrics.groupby(['evaluation_domain', 'task']))
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, max(4, 3.5 * len(panels))), squeeze=False)
    for ax, ((domain, task), panel) in zip(axes[:, 0], panels):
        labels = []
        for index, ((model, strategy, hierarchy_weight), values) in enumerate(
            panel.groupby(['model', 'strategy', 'hierarchy_loss_weight'])
        ):
            y = values['macro_f1'].to_numpy()
            jitter = np.linspace(-0.08, 0.08, len(y)) if len(y) > 1 else np.array([0.0])
            ax.scatter(index + jitter, y, alpha=0.65, s=28)
            mean = y.mean()
            ci = student_t.ppf(0.975, len(y) - 1) * y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.nan
            ax.errorbar(index, mean, yerr=ci, fmt='o', color='black', capsize=4)
            labels.append(f'{model}\\n{strategy}\\nh={hierarchy_weight:g}')
        if (domain, task) in chance:
            ax.axhline(chance[(domain, task)], color='0.35', linestyle='--', linewidth=1, label='Uniform-prediction expected macro-F1')
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha='right')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Macro-F1')
        ax.set_title(f'{domain} — {task}')
        ax.grid(axis='y', alpha=0.2)
        if (domain, task) in chance: ax.legend(fontsize=8)
    fig.tight_layout()
    for suffix in config['reporting']['formats']:
        fig.savefig(FIGURE_ROOT / f'final_model_comparison.{suffix}', bbox_inches='tight')
    plt.show()
else:
    print('Pending: final model comparisons require completed runs.')
"""),
        markdown("## Sequential transfer and forgetting"),
        code("""if not metrics.empty:
    sequential = metrics.loc[metrics['strategy'].isin(['peti_to_gbif', 'gbif_to_peti'])].copy()
    wide = sequential.pivot_table(
        index=['phase', 'model', 'strategy', 'hierarchy_loss_weight', 'seed', 'evaluation_domain', 'task'],
        columns='stage', values='macro_f1', aggfunc='first'
    ).reset_index()
    if {'stage1', 'stage2'}.issubset(wide.columns):
        wide['stage2_minus_stage1'] = wide['stage2'] - wide['stage1']
        wide.to_csv(REPORT_ROOT / 'transfer_forgetting_deltas.csv', index=False)
        display(wide.groupby(['phase', 'model', 'strategy', 'hierarchy_loss_weight', 'evaluation_domain', 'task'])['stage2_minus_stage1'].agg(['mean', 'std', 'count']))
    else:
        print('Pending: both sequential stages are required for transfer deltas.')
else:
    print('Pending: sequential metrics are not available.')
"""),
        markdown("## Learning curves and early stopping"),
        code("""history_rows = []
for history_path in sorted((OUTPUT_ROOT / 'runs').glob('*/*/seed-*/*/**/history.jsonl')):
    spec_path = history_path.parent / 'spec.json'
    if not spec_path.is_file(): continue
    spec = json.loads(spec_path.read_text())
    with history_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if 'validation' not in row: continue
            history_rows.append({
                **{k: spec.get(k, '') for k in ('phase', 'model', 'strategy', 'regime', 'stage')},
                'hierarchy_loss_weight': float(spec.get('hierarchy_loss_weight', spec.get('hierarchy_loss', {}).get('weight', 0.0))),
                'global_step': row['global_step'], 'stage_step': row['stage_step'],
                'validation_score': row['validation']['domain_balanced_macro_f1'],
                'train_loss': row['train_loss'],
            })
history = pd.DataFrame(history_rows)
if not history.empty:
    history.to_csv(REPORT_ROOT / 'validation_history.csv', index=False)
    fig, ax = plt.subplots(figsize=(11, 6))
        for keys, values in history.groupby(['phase', 'model', 'strategy', 'hierarchy_loss_weight']):
        curve = values.groupby('global_step')['validation_score'].mean()
            ax.plot(curve.index, curve.values, label=' / '.join(map(str, keys)))
    ax.set_xlabel('Optimizer step')
    ax.set_ylabel('Domain-balanced validation macro-F1')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for suffix in config['reporting']['formats']:
        fig.savefig(FIGURE_ROOT / f'learning_curves.{suffix}', bbox_inches='tight')
    plt.show()
else:
    print('Pending: validation histories are not available.')
"""),
        markdown("## Reproducibility record"),
        code("""def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
record = {
    'config': str(CONFIG_PATH),
    'config_sha256': sha256(CONFIG_PATH),
    'prepared_summary': prepared_summary,
    'inference_summary': inference_summary,
    'completed_metric_rows': int(len(metrics)),
    'completed_final_models': int(final_metrics[['phase','model','seed','strategy','hierarchy_loss_weight']].drop_duplicates().shape[0]) if not final_metrics.empty else 0,
}
(REPORT_ROOT / 'provenance.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\\n')
display(pd.Series({k: v for k, v in record.items() if not isinstance(v, dict)}, name='value').to_frame())
"""),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gbif_training.yaml")
    parser.add_argument(
        "--output", default="notebooks/gbif_inference_training_dino_results.ipynb"
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    build_notebook(args.config, output)
    print(f"Wrote {output}")
    if args.execute:
        with tempfile.TemporaryDirectory(prefix="gbif-notebook-") as temp_dir:
            environment = os.environ.copy()
            environment.update({
                "MPLCONFIGDIR": str(Path(temp_dir) / "matplotlib"),
                "JUPYTER_CONFIG_DIR": str(Path(temp_dir) / "jupyter-config"),
                "JUPYTER_DATA_DIR": str(Path(temp_dir) / "jupyter-data"),
            })
            subprocess.run([
                "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                str(output), "--ExecutePreprocessor.timeout=-1",
            ], check=True, env=environment)
        print(f"Executed {output}")


if __name__ == "__main__":
    main()
