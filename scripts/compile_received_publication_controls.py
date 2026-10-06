#!/usr/bin/env python3
"""Read local prediction artifacts, including staging; never load model weights.

Write a separate, timestamped partial report. No SSH, rsync, training, inference,
imports into the final run directory, or manuscript changes occur.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

TEST_SPLIT_PATH = None
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from worm_species.deployment_audit import f1_fixed


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_predictions(job, config, source):
    """Verify only small, fully received artifacts; a checkpoint is unnecessary."""
    receipt_path = source / 'audit_complete.json'
    receipt = read(receipt_path)
    if digest(Path(job['config'])) != job['config_sha256']:
        raise ValueError('Frozen configuration changed')
    if receipt['config_sha256'] != job['config_sha256']:
        raise ValueError('Receipt/config mismatch')
    paths = list(source.glob('*/test_predictions_best.csv'))
    if len(paths) != 1:
        raise ValueError('Prediction CSV not fully received')
    prediction_path = paths[0]
    run = prediction_path.parent
    summary_path = run / 'run_summary.json'
    map_path = run / 'label_to_index_by_task.json'
    summary, maps = read(summary_path), read(map_path)
    if summary != receipt['result'] or maps != config['audit_expected_label_maps']:
        raise ValueError('Saved summary/labels disagree with receipt/config')
    log_record = receipt.get('runtime_log')
    if receipt.get('runtime_path_mapping') and not log_record:
        raise ValueError('Genome completion log not finalised')
    sources = [receipt_path, prediction_path, summary_path, map_path, Path(job['config'])]
    if log_record:
        relative = Path(log_record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe log path')
        log = source / relative
        if digest(log) != log_record['sha256']:
            raise ValueError('Training log incomplete or changed')
        sources.append(log)
    before = {str(p): digest(p) for p in sources}
    frame = pd.read_csv(prediction_path, dtype={'individual_id': str, 'filename': str})
    test = pd.read_csv(TEST_SPLIT_PATH or Path(config['split']['predefined_split_dir']) / 'split_csv/test_split.csv',
                       dtype={'barcode': str, 'filename': str})
    if set(frame.task) != set(maps):
        raise ValueError('Incomplete prediction tasks')
    units = []
    for task, mapping in maps.items():
        g = frame[frame.task.eq(task)].copy()
        expected = test[test[config['data']['target_cols'][task]].isin(mapping)]
        if g.duplicated(['individual_id', 'filename']).any():
            raise ValueError('Duplicate image predictions')
        truth = expected.set_index(['barcode', 'filename'])[config['data']['target_cols'][task]]
        keys = list(zip(g.individual_id, g.filename))
        if set(keys) != set(truth.index):
            raise ValueError('Prediction/test population mismatch')
        if not all(truth.loc[key] == label for key, label in zip(keys, g.true_label)):
            raise ValueError('Truth differs from fixed test metadata')
        names = sorted(mapping, key=mapping.get)
        encoded = [json.loads(x) for x in g.class_probabilities_json]
        if not all(set(p) == set(mapping) for p in encoded):
            raise ValueError('Incomplete probability vocabulary')
        probabilities = np.asarray([[p[name] for name in names] for p in encoded], float)
        if not (np.isfinite(probabilities).all() and (probabilities >= 0).all()
                and (probabilities <= 1).all() and np.allclose(probabilities.sum(1), 1, atol=.002)):
            raise ValueError('Invalid saved probabilities')
        y, predicted = g.true_index.to_numpy(int), g.predicted_index.to_numpy(int)
        if not (np.isin(y, list(mapping.values())).all()
                and np.isin(predicted, list(mapping.values())).all()):
            raise ValueError('Out-of-vocabulary indices')
        if not all(mapping[name] == i for name, i in zip(g.true_label, y)):
            raise ValueError('Truth labels/indices disagree')
        if not np.allclose(probabilities[np.arange(len(g)), predicted], probabilities.max(1), atol=1e-7):
            raise ValueError('Saved image prediction is not maximal')
        if (summary[f'test_{task}_n'] != len(g)
                or not np.isclose(f1_score(y, predicted, average='macro', zero_division=0),
                                  summary[f'test_{task}_macro_f1'], atol=1e-10)):
            raise ValueError('Image counts/F1 disagree with saved summary')
        for worm, positions in g.groupby('individual_id', sort=False).indices.items():
            if len(set(y[positions])) != 1:
                raise ValueError('Inconsistent within-worm label')
            units.append(dict(condition=job['condition'], seed=job['seed'], kind=job['kind'],
                              task=task, individual_id=worm, true_index=int(y[positions[0]]),
                              predicted_index=int(probabilities[positions].mean(0).argmax()),
                              n_images=len(positions), source=str(prediction_path)))
    if any(digest(Path(p)) != checksum for p, checksum in before.items()):
        raise ValueError('Source changed during reading; retry on next snapshot')
    return units, before


def plot_excess_loss(summary, out, complete):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout='constrained')
    for ax, task in zip(axes, ['genus', 'species']):
        q = summary[summary.task.eq(task) & summary.cohort.eq('full_test')]
        ax.scatter(range(len(q)), q['mean'], color='#176b91')
        good = q.seed_sd.notna().to_numpy()
        ax.errorbar(np.arange(len(q))[good], q['mean'].to_numpy()[good],
                    yerr=q.seed_sd.to_numpy()[good], fmt='none', capsize=3, color='#176b91')
        ax.axhline(0, color='black', linewidth=.8)
        labels = [f"{r.species.replace('_', ' ')}\n{r.stage_removed}; n={r.seeds} seeds" for r in q.itertuples()]
        ax.set_xticks(range(len(q)), labels, rotation=35, ha='right', fontsize=8)
        ax.set_title(task.capitalize() + ' target-class F1')
    axes[0].set_ylabel('Additional loss beyond matched random removal (pp)')
    fig.suptitle('Complete 30-seed matched-control results' if complete else 'PARTIAL snapshot: complete paired comparisons only', fontweight='bold')
    fig.savefig(out / 'coverage_excess_loss.png', dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='dev/publication_deployment_audit_v2.yaml')
    parser.add_argument('--csv-root', type=Path, help='Separate CSV-only download directory; default: output/csv_only/runs')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = ROOT / cfg['output']
    plan = read(root / 'training/plan.json')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out = root / 'received_csv_summary' / stamp
    out.mkdir(parents=True)
    final = Path(cfg['training_run_root'])
    stage = final.parent / 'genome_download/runs'
    csv_root = args.csv_root or root / 'csv_only/runs'
    # Freeze the candidate list before scoring; newly arriving files wait for rerun.
    candidates = []
    for job in plan['jobs']:
        config_path=Path(job['config'])
        if not config_path.is_file():
            config_path=root/'training/configs'/job['condition']/f"seed_{job['seed']}.json"
        job=dict(job,config=str(config_path))
        config = read(config_path)
        for origin, directory in [('final', Path(config['output']['out_dir'])),
                                  ('csv_only', csv_root / job['condition'] / f"seed_{job['seed']}"),
                                  ('staging', stage / job['condition'] / f"seed_{job['seed']}")]:
            if (directory / 'audit_complete.json').is_file():
                candidates.append((job, config, origin, directory))
    units, inventory, sources, accepted = [], [], {}, set()
    for job, config, origin, directory in candidates:
        if job['id'] in accepted:
            continue
        row = dict(job_id=job['id'], condition=job['condition'], seed=job['seed'],
                   kind=job['kind'], origin=origin, source=str(directory))
        try:
            rows, hashes = validate_predictions(job, config, directory)
            units.extend(rows); sources.update(hashes); accepted.add(job['id'])
            row.update(status='prediction_artifacts_verified', reason='')
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            row.update(status='not_scored', reason=str(exc))
        inventory.append(row)
        if len(accepted) and len(accepted) % 25 == 0:
            print(f'Verified prediction CSVs for {len(accepted)} fits', flush=True)
    pd.DataFrame(inventory).to_csv(out / 'artifact_inventory.csv', index=False)
    f = pd.DataFrame(units)
    f.to_csv(out / 'individual_predictions.csv.gz', index=False)
    if f.empty:
        raise RuntimeError('No complete local prediction artifacts; see inventory')
    complete = len(accepted) == len(plan['jobs'])
    conditions = {c['name']: c for c in plan['conditions']}
    maps = read(Path(plan['jobs'][0]['config']))['audit_expected_label_maps']
    macro = []
    for (name, seed, task), g in f.groupby(['condition', 'seed', 'task']):
        macro.append(dict(condition=name, seed=seed, kind=conditions[name]['kind'], task=task,
                          individuals=len(g), images=int(g.n_images.sum()),
                          individual_macro_f1=float(f1_fixed(g.true_index, g.predicted_index,
                                                            range(len(maps[task])), len(maps[task])))))
    macro = pd.DataFrame(macro); macro.to_csv(out / 'individual_macro_f1_per_seed.csv', index=False)
    macro.groupby(['condition', 'kind', 'task']).individual_macro_f1.agg(['mean', 'std', 'count']).reset_index().rename(
        columns={'std': 'seed_sd', 'count': 'seeds'}).to_csv(out / 'individual_macro_f1_summary.csv', index=False)
    full = f[f.kind.eq('reference')]
    scored = []
    for name, condition in conditions.items():
        if condition['kind'] == 'reference':
            continue
        test = pd.read_csv(root / 'training/splits' / name / 'split_csv/test_split.csv')
        target = set(test.loc[test.species_label.eq(condition['species']) & test.life_stage.eq(condition['stage']), 'barcode'])
        for (seed, task), g in f[f.condition.eq(name) & f.task.isin(['genus', 'species'])].groupby(['seed', 'task']):
            baseline = full[full.seed.eq(seed) & full.task.eq(task)]
            if baseline.empty:
                continue
            label = maps[task][condition['species'] if task == 'species' else condition['species'].split('_')[0]]
            for cohort, ids in [('full_test', set(g.individual_id)), ('removed_stage_test', target)]:
                x, y = g[g.individual_id.isin(ids)], baseline[baseline.individual_id.isin(ids)]
                if x.empty:
                    continue
                if set(x.individual_id) != set(y.individual_id):
                    raise ValueError('Reference/control individual mismatch')
                before = float(f1_fixed(y.true_index, y.predicted_index, [label], len(maps[task])))
                after = float(f1_fixed(x.true_index, x.predicted_index, [label], len(maps[task])))
                scored.append(dict(condition=name, kind=condition['kind'], species=condition['species'],
                                   stage_removed=condition['stage'], repeat=condition['repeat'], seed=seed,
                                   task=task, cohort=cohort, individuals=len(x), reference_class_f1=before,
                                   class_f1=after, loss_pp=100 * (before - after)))
    s = pd.DataFrame(scored); s.to_csv(out / 'control_per_seed.csv', index=False)
    excess = pd.DataFrame()
    if not s.empty:
        key = ['species', 'stage_removed', 'task', 'cohort', 'seed']
        random = s[s.kind.eq('matched_random')].groupby(key).agg(
            random_class_f1=('class_f1', 'mean'), completed_repeats=('repeat', 'nunique')).reset_index()
        random = random[random.completed_repeats.eq(cfg['training_control_repeats'])]
        targeted = s.loc[s.kind.eq('stage_removed'), key + ['class_f1', 'reference_class_f1']].rename(
            columns={'class_f1': 'stage_removed_class_f1'})
        excess = targeted.merge(random, on=key, validate='one_to_one')
        excess['excess_loss_pp'] = 100 * (excess.random_class_f1 - excess.stage_removed_class_f1)
    excess.to_csv(out / 'coverage_excess_loss_per_seed.csv', index=False)
    if not excess.empty:
        summary = excess.groupby(['species', 'stage_removed', 'task', 'cohort']).excess_loss_pp.agg(
            ['mean', 'std', 'count']).reset_index().rename(columns={'std': 'seed_sd', 'count': 'seeds'})
        summary.to_csv(out / 'coverage_excess_loss_summary.csv', index=False)
        plot_excess_loss(summary, out, complete)
    coverage = pd.DataFrame([dict(condition=c['name'], kind=c['kind'],
                                 received_fits=sum(j['id'] in accepted for j in plan['jobs'] if j['condition'] == c['name']),
                                 planned_fits=sum(j['condition'] == c['name'] for j in plan['jobs'])) for c in plan['conditions']])
    coverage.to_csv(out / 'condition_coverage.csv', index=False)
    status = dict(snapshot_utc=stamp, verified_prediction_fits=len(accepted), planned_fits=len(plan['jobs']),
                  source_final=sum(r['status']=='prediction_artifacts_verified' and r['origin']=='final' for r in inventory),
                  source_csv_only=sum(r['status']=='prediction_artifacts_verified' and r['origin']=='csv_only' for r in inventory),
                  source_staging=sum(r['status']=='prediction_artifacts_verified' and r['origin']=='staging' for r in inventory),
                  complete_pair_seed_groups=0 if excess.empty else len(excess[['species','stage_removed','seed']].drop_duplicates()),
                  weights_loaded=False, network_used=False, training_run=False,
                  complete_prediction_cohort=complete,
                  report_scope=('Complete planned prediction cohort' if complete else 'Partial received prediction artifacts') + '; no checkpoint or weight verification.')
    (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    (out / 'source_checksums.json').write_text(json.dumps(sources, indent=2) + '\n')
    (out.parent / 'latest.json').write_text(json.dumps(dict(path=str(out), **status), indent=2) + '\n')
    label = 'Complete' if complete else 'Partial'
    text = [f'# {label} received-results summary — {stamp}', '',
            f"Verified saved prediction artifacts: **{len(accepted)}/{len(plan['jobs'])} planned fits**.",
            f"Previously imported/local: {status['source_final']}; CSV-only folder: {status['source_csv_only']}; download staging: {status['source_staging']}.",
            '', 'No network connection, model-weight loading, fresh inference, training or final-directory import was performed.',
            'The background transfer is independent. This is a fixed snapshot; rerun the Make target for newly received CSVs.',
            '', 'Individual predictions average complete per-image probability vectors within each worm. Macro-F1 uses the fixed full output vocabulary.',
            'Biological comparisons report genus/species target-class F1, not full-task macro-F1.',
            'The main coverage contrast uses the entire independent test population, retaining false positives from other taxa.',
            'The removed-stage-only diagnostic cannot measure false positives from excluded test groups and is secondary.',
            'An excess-loss contrast requires the matching full reference, stage removal and all three random-removal repeats at the same neural seed.',
            'Positive excess loss means stage removal performs worse than equally sized random removal. Error bars, when available, are seed SD, not biological sampling uncertainty.',
            '', f"Complete species/stage/seed comparisons: **{status['complete_pair_seed_groups']}**.",
            ('All planned conditions contain 30 seeds; all six biological contrasts contain all three matched repeats per seed.' if complete else 'Partial arrival order is non-random; these summaries must not replace the full 30-seed analysis or update the manuscript yet.'),
            'Source hashes, validation exclusions and per-condition completion counts accompany the tables. Checkpoint integrity remains untested here.',
            '', '## Per-condition coverage', '']
    text += ['| Condition | Received | Planned |', '|---|---:|---:|']
    text += [f'| {r.condition} | {r.received_fits} | {r.planned_fits} |' for r in coverage.itertuples()]
    if not excess.empty:
        text += ['', '## Coverage loss beyond matched random removal', '',
                 '| Species | Removed stage | Task | Excess loss (pp) | Seed SD | Seeds |', '|---|---|---|---:|---:|---:|']
        for r in summary[summary.cohort.eq('full_test')].itertuples():
            sd = 'not estimable' if pd.isna(r.seed_sd) else f'{r.seed_sd:.2f}'
            text.append(f'| {r.species.replace("_", " ")} | {r.stage_removed} | {r.task} | {r.mean:+.2f} | {sd} | {r.seeds} |')
    (out / 'README.md').write_text('\n'.join(text) + '\n')
    print(json.dumps(dict(path=str(out), **status), indent=2))


if __name__ == '__main__':
    main()
