#!/usr/bin/env python3
"""Build a read-only notebook for full-taxonomy run completion and results."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


DEFAULT_OUTPUT = Path("notebooks/gbif_full_taxonomy_run_status.ipynb")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source)


def code(source: str):
    return nbf.v4.new_code_cell(source)


def build_notebook(config_path: str, output_path: Path) -> None:
    setup = """from pathlib import Path
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

CONFIG_VALUE = Path(__CONFIG_PATH__)
config_candidates = [CONFIG_VALUE, Path.cwd() / CONFIG_VALUE, Path.cwd().parent / CONFIG_VALUE]
CONFIG_PATH = next((path.resolve() for path in config_candidates if path.is_file()), None)
if CONFIG_PATH is None:
    raise FileNotFoundError(f'Could not resolve full-taxonomy config from: {config_candidates}')
PROJECT_ROOT = CONFIG_PATH.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from scripts.gbif_full_taxonomy_pipeline import build_specs

def expand(value):
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    return value

config = expand(yaml.safe_load(CONFIG_PATH.read_text()))
configured_root = Path(config['paths']['experiment_root'])
OUTPUT_ROOT = Path(os.environ.get('GBIF_FULL_TAXONOMY_ROOT', configured_root))
config['paths']['experiment_root'] = str(OUTPUT_ROOT)
STATUS_OUTPUT_ROOT = Path(os.environ.get(
    'GBIF_FULL_TAXONOMY_STATUS_OUTPUT',
    PROJECT_ROOT / 'outputs' / 'gbif_full_taxonomy_status',
))
STATUS_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
specs_by_phase = build_specs(config)
print('Configuration:', CONFIG_PATH)
print('Full-taxonomy results:', OUTPUT_ROOT)
print('Status outputs:', STATUS_OUTPUT_ROOT)
print('This notebook reads artifacts only; it does not query Slurm, train, or submit jobs.')
""".replace("__CONFIG_PATH__", repr(config_path))

    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "wormspecies", "language": "python", "name": "python3"
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    notebook["cells"] = [
        markdown(
            "# Full-taxonomy run status and completed results\n\n"
            "This notebook shows the file-backed state of every planned full-taxonomy run: "
            "9 Petri pretraining runs, 18 primary GBIF runs without hierarchy loss, and 18 "
            "revised-hierarchy GBIF runs. It also summarizes completed test metrics, inference "
            "outputs, and final-report availability.\n\n"
            "**Completion contract:** a training run is complete only when `run_status.json` "
            "says `complete` and `best_model.pt` exists. `last_model.pt` marks an interrupted "
            "run as resumable. This notebook does not infer live Slurm state and cannot submit jobs."
        ),
        code(setup),
        markdown("## Discover every planned training run"),
        code("""def read_json(path):
    try:
        return json.loads(path.read_text()) if path.is_file() else None
    except (OSError, json.JSONDecodeError):
        return {'_read_error': True}

def last_history_record(path):
    if not path.is_file():
        return None
    try:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        return json.loads(lines[-1]) if lines else None
    except (OSError, json.JSONDecodeError):
        return None

training_rows = []
test_metric_rows = []
for phase in ('petri', 'primary', 'hierarchy'):
    for array_index, spec in enumerate(specs_by_phase[phase]):
        output = Path(spec['output_dir'])
        status_path = output / 'run_status.json'
        best_path = output / 'best_model.pt'
        last_path = output / 'last_model.pt'
        history_path = output / 'history.jsonl'
        test_path = output / 'test_metrics.json'
        status = read_json(status_path)
        history = last_history_record(history_path)
        status_complete = bool(status and status.get('status') == 'complete')
        if status_complete and best_path.is_file():
            artifact_status = 'complete'
        elif status_complete and not best_path.is_file():
            artifact_status = 'inconsistent_missing_best'
        elif last_path.is_file():
            artifact_status = 'resumable'
        elif best_path.is_file() or history_path.is_file() or output.is_dir():
            artifact_status = 'started'
        else:
            artifact_status = 'pending'
        last_step = history.get('step') if history else None
        row = {
            'phase': phase,
            'array_index': array_index,
            'run_id': spec['run_id'],
            'model': spec['model'],
            'seed': int(spec['seed']),
            'domain': spec['domain'],
            'condition': spec['condition'],
            'hierarchy_kind': spec['hierarchy_kind'],
            'hierarchy_loss_weight': float(spec.get('hierarchy_loss_weight', 0.0)),
            'artifact_status': artifact_status,
            'last_step': last_step,
            'max_steps': int(spec['max_steps']),
            'progress_fraction': (
                min(float(last_step) / int(spec['max_steps']), 1.0)
                if last_step is not None else (1.0 if artifact_status == 'complete' else 0.0)
            ),
            'best_checkpoint': best_path.is_file(),
            'last_checkpoint': last_path.is_file(),
            'history': history_path.is_file(),
            'best_validation_species_macro_f1': (
                status.get('best_validation_species_macro_f1') if status else None
            ),
            'best_validation_genus_macro_f1': (
                status.get('best_validation_genus_macro_f1') if status else None
            ),
            'output_dir': str(output),
        }
        training_rows.append(row)
        test_metrics = read_json(test_path)
        if artifact_status == 'complete' and test_metrics and not test_metrics.get('_read_error'):
            metric_row = {key: row[key] for key in (
                'phase', 'array_index', 'run_id', 'model', 'seed', 'domain',
                'condition', 'hierarchy_kind', 'hierarchy_loss_weight'
            )}
            metric_row['test_loss'] = test_metrics.get('loss')
            for task in ('genus', 'species', 'age'):
                values = test_metrics.get(task, {})
                for metric in ('n', 'accuracy', 'balanced_accuracy', 'macro_f1'):
                    metric_row[f'{task}_{metric}'] = values.get(metric)
            test_metric_rows.append(metric_row)

training_status = pd.DataFrame(training_rows)
test_metrics = pd.DataFrame(test_metric_rows)
training_status.to_csv(STATUS_OUTPUT_ROOT / 'training_run_status.csv', index=False)
test_metrics.to_csv(STATUS_OUTPUT_ROOT / 'completed_test_metrics.csv', index=False)
display(training_status.head())
"""),
        markdown("## Completion summary"),
        code("""status_order = [
    'complete', 'resumable', 'started', 'inconsistent_missing_best', 'pending'
]
completion_summary = pd.crosstab(
    training_status['phase'], training_status['artifact_status']
).reindex(index=['petri', 'primary', 'hierarchy'], fill_value=0)
completion_summary = completion_summary.reindex(columns=status_order, fill_value=0)
completion_summary['planned'] = completion_summary.sum(axis=1)
completion_summary['completion_fraction'] = (
    completion_summary['complete'] / completion_summary['planned']
)
display(completion_summary)
completion_summary.to_csv(STATUS_OUTPUT_ROOT / 'training_completion_summary.csv')

incomplete = training_status.loc[training_status['artifact_status'].ne('complete')]
incomplete_indices = {
    phase: incomplete.loc[incomplete['phase'].eq(phase), 'array_index'].astype(int).tolist()
    for phase in ('petri', 'primary', 'hierarchy')
}
print('Incomplete array indices:', json.dumps(incomplete_indices, sort_keys=True))

colors = {
    'complete': '#2F6B5F', 'resumable': '#D9A441', 'started': '#477998',
    'inconsistent_missing_best': '#A44A3F', 'pending': '#B8B8B8',
}
plot_frame = completion_summary[status_order]
ax = plot_frame.plot.bar(
    stacked=True, figsize=(10, 5), color=[colors[column] for column in status_order]
)
ax.set(title='Full-taxonomy training artifact status', xlabel='Phase', ylabel='Runs')
ax.legend(title='Status', bbox_to_anchor=(1.02, 1), loc='upper left')
ax.grid(axis='y', alpha=.2)
plt.tight_layout()
plt.savefig(STATUS_OUTPUT_ROOT / 'training_completion.png', dpi=180, bbox_inches='tight')
plt.show()
"""),
        markdown("## Run-level status and resumable progress"),
        code("""display_columns = [
    'phase', 'array_index', 'run_id', 'model', 'seed', 'condition',
    'hierarchy_kind', 'artifact_status', 'last_step', 'max_steps',
    'progress_fraction', 'best_validation_species_macro_f1',
    'best_validation_genus_macro_f1',
]
display(training_status[display_columns])

active_progress = training_status.loc[
    training_status['artifact_status'].isin(['resumable', 'started'])
].sort_values(['phase', 'array_index'])
if active_progress.empty:
    print('No interrupted or partially started training runs were found.')
else:
    display(active_progress[display_columns])
"""),
        markdown("## Metrics from completed runs"),
        code("""if test_metrics.empty:
    print('No completed test_metrics.json artifacts are available yet.')
else:
    metric_columns = [
        column for column in (
            'phase', 'run_id', 'model', 'seed', 'condition', 'hierarchy_kind',
            'genus_balanced_accuracy', 'genus_macro_f1',
            'species_balanced_accuracy', 'species_macro_f1',
            'age_balanced_accuracy', 'age_macro_f1',
        ) if column in test_metrics
    ]
    display(test_metrics[metric_columns].sort_values(['phase', 'model', 'seed', 'condition']))
    gbif_metrics = test_metrics.loc[test_metrics['domain'].eq('gbif')].copy()
    if not gbif_metrics.empty and 'species_macro_f1' in gbif_metrics:
        gbif_metrics['series'] = (
            gbif_metrics['condition'].astype(str) + ' | ' +
            gbif_metrics['hierarchy_kind'].astype(str)
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        for offset, (series, group) in enumerate(gbif_metrics.groupby('series', sort=True)):
            x = np.arange(len(group)) + offset * .03
            ax.scatter(x, group['species_macro_f1'], label=series, alpha=.8, s=45)
        ax.set(
            title='Completed GBIF test species macro-F1', xlabel='Completed run',
            ylabel='Species macro-F1', ylim=(0, 1)
        )
        ax.grid(axis='y', alpha=.2)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(STATUS_OUTPUT_ROOT / 'completed_species_macro_f1.png', dpi=180, bbox_inches='tight')
        plt.show()
"""),
        markdown("## Inference and final-report status"),
        code("""final_specs = specs_by_phase['primary'] + specs_by_phase['hierarchy']
inference_rows = []
for array_index, spec in enumerate(final_specs):
    checkpoint = Path(spec['output_dir']) / 'best_model.pt'
    output = OUTPUT_ROOT / 'inference' / f"{spec['run_id']}.csv.gz"
    summary_path = output.with_suffix('').with_suffix('.summary.json')
    summary = read_json(summary_path)
    complete = bool(
        checkpoint.is_file() and output.is_file() and summary
        and summary.get('status') == 'complete'
    )
    inference_rows.append({
        'array_index': array_index,
        'run_id': spec['run_id'],
        'model': spec['model'],
        'seed': int(spec['seed']),
        'condition': spec['condition'],
        'hierarchy_kind': spec['hierarchy_kind'],
        'status': 'complete' if complete else 'pending',
        'rows': summary.get('rows') if summary else None,
        'checkpoint_exists': checkpoint.is_file(),
        'predictions_exist': output.is_file(),
        'summary_exists': summary_path.is_file(),
        'output': str(output),
    })
inference_status = pd.DataFrame(inference_rows)
inference_status.to_csv(STATUS_OUTPUT_ROOT / 'inference_status.csv', index=False)
display(inference_status['status'].value_counts().to_frame('tasks'))
display(inference_status)

final_manifest_path = OUTPUT_ROOT / 'final_manifest.json'
final_report_path = OUTPUT_ROOT / 'final_report.md'
final_manifest = read_json(final_manifest_path)
final_report_complete = bool(
    final_manifest and final_manifest.get('status') == 'complete' and final_report_path.is_file()
)
print('Final report:', 'complete' if final_report_complete else 'pending')
if final_manifest:
    display(pd.Series(final_manifest, name='value').to_frame())
"""),
        markdown("## Submission receipts and machine-readable summary"),
        code("""receipt_rows = []
for name in ('submission_receipt.json', 'resume_submission_receipt.json'):
    path = OUTPUT_ROOT / 'generated' / name
    payload = read_json(path)
    if payload:
        receipt_rows.append({'receipt': name, **payload})
if receipt_rows:
    display(pd.DataFrame(receipt_rows))
else:
    print('No submission receipts were found in this output tree.')

status_summary = {
    'output_root': str(OUTPUT_ROOT),
    'training_planned': int(len(training_status)),
    'training_complete': int(training_status['artifact_status'].eq('complete').sum()),
    'training_resumable': int(training_status['artifact_status'].eq('resumable').sum()),
    'training_incomplete_indices': incomplete_indices,
    'inference_planned': int(len(inference_status)),
    'inference_complete': int(inference_status['status'].eq('complete').sum()),
    'final_report_complete': final_report_complete,
}
(STATUS_OUTPUT_ROOT / 'status_summary.json').write_text(
    json.dumps(status_summary, indent=2, sort_keys=True) + '\\n'
)
display(pd.Series(status_summary, name='value').to_frame())
print('Saved status tables and figures:', STATUS_OUTPUT_ROOT)
"""),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gbif_full_taxonomy.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_notebook(args.config, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
