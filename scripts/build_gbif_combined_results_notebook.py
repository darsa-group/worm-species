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
            "# GBIF–Petri inference and domain-order results\n\n"
            "This notebook combines the sharded baseline inference, the five-seed "
            "ViT/ResNet/ConvNeXt experiments, and the later three-seed DINOv3 "
            "experiment. It reads completed artifacts only and never trains or submits jobs."
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
        markdown("## Baseline inference on curated GBIF"),
        code("""inference_path = OUTPUT_ROOT / 'inference' / 'baseline' / 'predictions.csv'
inference_summary_path = inference_path.with_suffix('.summary.json')
if inference_path.is_file() and inference_summary_path.is_file():
    inference = pd.read_csv(inference_path, dtype=str, keep_default_na=False)
    inference_summary = json.loads(inference_summary_path.read_text())
    display(pd.Series(inference_summary, name='value').to_frame())
    scope = pd.DataFrame({
        'genus': inference['checkpoint_genus_scope'].value_counts(),
        'species': inference['checkpoint_species_scope'].value_counts(),
    }).fillna(0).astype(int)
    display(scope)
else:
    inference = pd.DataFrame()
    inference_summary = None
    print('Pending: merged 12-shard baseline inference is not available.')
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
                **{k: spec[k] for k in ('phase', 'model', 'seed', 'regime', 'stage', 'domain')},
                'evaluation_domain': domain, 'task': task,
                'macro_f1': float(value), 'n': int(n),
                'final_model': bool(spec.get('final_model', False)),
                'stopped_early': bool(status.get('stopped_early', False)),
                'completed_steps': int(status.get('stage_step', 0)),
            })
metrics = pd.DataFrame(rows)
print(f'Collected {{len(metrics)}} completed domain-task rows.')
display(metrics.head()) if not metrics.empty else print('Pending: no completed training stages.')
if not metrics.empty:
    metrics.to_csv(REPORT_ROOT / 'all_stage_metrics.csv', index=False)
"""),
        markdown("## Final model comparison"),
        code("""final_metrics = metrics.loc[metrics['final_model']].copy() if not metrics.empty else pd.DataFrame()
if not final_metrics.empty:
    summary = final_metrics.groupby(
        ['phase', 'model', 'regime', 'evaluation_domain', 'task'], as_index=False
    ).agg(mean_macro_f1=('macro_f1', 'mean'), sd=('macro_f1', 'std'), seeds=('seed', 'nunique'))
    summary['ci95'] = student_t.ppf(0.975, summary['seeds'] - 1) * summary['sd'] / np.sqrt(summary['seeds'])
    summary.to_csv(REPORT_ROOT / 'final_model_summary.csv', index=False)
    display(summary)

    label_maps_path = OUTPUT_ROOT / 'prepared' / 'label_maps.json'
    label_maps = json.loads(label_maps_path.read_text())
    chance = {}
    for domain in ('gbif', 'petri'):
        test_path = OUTPUT_ROOT / 'prepared' / f'{{domain}}_test.csv'
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
        for index, ((model, regime), values) in enumerate(panel.groupby(['model', 'regime'])):
            y = values['macro_f1'].to_numpy()
            jitter = np.linspace(-0.08, 0.08, len(y)) if len(y) > 1 else np.array([0.0])
            ax.scatter(index + jitter, y, alpha=0.65, s=28)
            mean = y.mean()
            ci = student_t.ppf(0.975, len(y) - 1) * y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.nan
            ax.errorbar(index, mean, yerr=ci, fmt='o', color='black', capsize=4)
            labels.append(f'{{model}}\\n{{regime}}')
        if (domain, task) in chance:
            ax.axhline(chance[(domain, task)], color='0.35', linestyle='--', linewidth=1, label='Uniform-prediction expected macro-F1')
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha='right')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Macro-F1')
        ax.set_title(f'{{domain}} — {{task}}')
        ax.grid(axis='y', alpha=0.2)
        if (domain, task) in chance: ax.legend(fontsize=8)
    fig.tight_layout()
    for suffix in config['reporting']['formats']:
        fig.savefig(FIGURE_ROOT / f'final_model_comparison.{{suffix}}', bbox_inches='tight')
    plt.show()
else:
    print('Pending: final model comparisons require completed runs.')
"""),
        markdown("## Sequential transfer and forgetting"),
        code("""if not metrics.empty:
    sequential = metrics.loc[metrics['regime'].isin(['curated_then_petri', 'petri_then_curated'])].copy()
    wide = sequential.pivot_table(
        index=['phase', 'model', 'seed', 'regime', 'evaluation_domain', 'task'],
        columns='stage', values='macro_f1', aggfunc='first'
    ).reset_index()
    if {'stage1', 'stage2'}.issubset(wide.columns):
        wide['stage2_minus_stage1'] = wide['stage2'] - wide['stage1']
        wide.to_csv(REPORT_ROOT / 'transfer_forgetting_deltas.csv', index=False)
        display(wide.groupby(['phase', 'model', 'regime', 'evaluation_domain', 'task'])['stage2_minus_stage1'].agg(['mean', 'std', 'count']))
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
                **{k: spec[k] for k in ('phase', 'model', 'seed', 'regime', 'stage')},
                'global_step': row['global_step'], 'stage_step': row['stage_step'],
                'validation_score': row['validation']['domain_balanced_macro_f1'],
                'train_loss': row['train_loss'],
            })
history = pd.DataFrame(history_rows)
if not history.empty:
    history.to_csv(REPORT_ROOT / 'validation_history.csv', index=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    for keys, values in history.groupby(['phase', 'model', 'regime']):
        curve = values.groupby('global_step')['validation_score'].mean()
        ax.plot(curve.index, curve.values, label=' / '.join(keys))
    ax.set_xlabel('Optimizer step')
    ax.set_ylabel('Domain-balanced validation macro-F1')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for suffix in config['reporting']['formats']:
        fig.savefig(FIGURE_ROOT / f'learning_curves.{{suffix}}', bbox_inches='tight')
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
    'completed_final_models': int(final_metrics[['phase','model','seed','regime']].drop_duplicates().shape[0]) if not final_metrics.empty else 0,
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
