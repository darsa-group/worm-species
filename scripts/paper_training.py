#!/usr/bin/env python3
"""Frozen training plans with the artifact contracts consumed by scripts/paper.py.

No scheduler or remote-service dependency. All new fits have a separate root.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import pandas as pd
import yaml
from scripts import paper
from scripts import run_publication_matched_controls as controls
from scripts.run_publication_external_test import lock
from worm_species.config.normalization import normalize_config
from worm_species.config.sweeps import generate_sweep_configs
from worm_species.data.labels import build_label_maps

STAGES = ('baseline', 'visual_ablation', 'visual_interactions',
          'adult_taxon_baseline', 'adult_taxon_holdouts')
REQUIRED = ('best_model.pt', 'config.json', 'label_to_index_by_task.json',
            'test_predictions_best.csv', 'run_summary.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    controls.save(Path(path), obj)


def settings(paper_config, training_config):
    cfg = paper.load_settings(paper_config)
    train = yaml.safe_load(Path(training_config).read_text())
    root = (ROOT / Path(train['output_root']).expanduser()).resolve()
    # Reference artifacts and all existing analysis folders are protected.
    for value in cfg['paths'].values():
        original = Path(value)
        if root == original or root in original.parents or original in root.parents:
            raise ValueError(f'New training root must be separate from existing inputs/outputs: {root} / {original}')
    if train['workers'] < 0 or train['cpu_threads'] < 1:
        raise ValueError('Invalid training worker/thread count')
    for key in ('batch_size', 'control_batch_size'):
        if train.get(key) is not None and train[key] < 1:
            raise ValueError('Training batches must be positive')
    return cfg, train, root


def prepared_splits(cfg):
    """Map the frozen reference cohort to processed images, without resplitting."""
    dataset = Path(cfg['paths']['dataset'])
    raw = pd.read_csv(dataset / 'metadata/images.csv')
    old = raw.loc[raw.year.eq(2025) & raw.kind.eq('worm')]
    if old.duplicated(['barcode', 'filename']).any():
        raise ValueError('Ambiguous reference image metadata')
    lookup = old.set_index(['barcode', 'filename'])
    frames = {}
    for name in ('train', 'val', 'test'):
        frame = pd.read_csv(Path(cfg['paths']['splits']) / f'{name}_split.csv')
        if frame.duplicated(['barcode', 'filename']).any():
            raise ValueError(f'Duplicate {name} image')
        for column in ('rel_path_raw', 'rel_path_seg', 'rel_path_segmask'):
            frame[column] = [lookup.loc[(r.barcode, r.filename), column] for r in frame.itertuples()]
        for rel in frame.rel_path_seg:
            path = dataset / str(rel)
            if pd.isna(rel) or not path.resolve().is_relative_to(dataset) or not path.is_file():
                raise FileNotFoundError(f'Missing/unsafe prepared image: {rel}')
        frames[name] = frame
    ids = {k: set(v.barcode) for k, v in frames.items()}
    if any(ids[a] & ids[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('Worm occurs in multiple reference splits')
    return frames


def sources(cfg, train):
    paths = [ROOT/'scripts/paper_training.py', ROOT/'scripts/run_publication_matched_controls.py',
             Path(cfg['paths']['dataset'])/'metadata/images.csv']
    paths += [Path(cfg['paths']['splits'])/f'{n}_split.csv' for n in ('train', 'val', 'test')]
    paths += list((ROOT/'configs/training').glob('*.yaml'))
    paths += list((ROOT/'src').rglob('*.py'))
    return {'settings': train, 'source_sha256': {str(p):sha(p) for p in sorted(paths)}}


def prepare(cfg, train, root):
    root.mkdir(parents=True, exist_ok=True)
    identity = sources(cfg, train)
    manifest = root/'training_plan.json'
    if manifest.exists():
        plan = json.loads(manifest.read_text())
        if plan['identity'] != identity:
            raise ValueError('Training settings/sources changed; choose a fresh output root')
        validate_plan(plan)
        return plan
    if any(root.iterdir()):
        raise ValueError('Training root is not empty and has no frozen plan; choose a fresh root')
    frames = prepared_splits(cfg)
    for name, frame in frames.items():
        target = root/'splits/split_csv'/f'{name}_split.csv'
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
    metadata = root/'splits/metadata.csv'
    pd.concat(frames.values(), ignore_index=True).to_csv(metadata, index=False)
    jobs, baseline = [], []
    for stage in STAGES:
        base = normalize_config(yaml.safe_load((ROOT/'configs/training'/f'{stage}.yaml').read_text()))
        for i, config in enumerate(generate_sweep_configs(base)):
            config['sweep']['enabled'] = False
            config['data'].update(root_dir=cfg['paths']['dataset'], metadata_csv=str(metadata))
            config['split'].update(predefined_split_dir=str(root/'splits'), use_predefined_splits=True, save_splits=False)
            config['training'].update(device=train['device'], num_workers=train['workers'], cpu_threads=train['cpu_threads'])
            if train.get('batch_size') is not None:
                config['training']['batch_size'] = train['batch_size']
            config['cache'] = {'enabled':False, 'condition_variants':{'enabled':False}}
            config['wandb'].update(train['wandb'])
            directory = root/'reference/runs'/stage/f'fit_{i:04d}'
            config['output']['out_dir'] = str(directory)
            path = root/'configs'/stage/f'{i:04d}.json'
            save(path, config)
            jobs.append(dict(stage=stage, index=i, config=str(path), config_sha256=sha(path), output=str(directory)))
            if stage=='baseline' and config['model']['name']=='convnext_base':
                maps, _ = build_label_maps(frames['train'], config['data']['target_cols'])
                baseline.append(dict(model='convnext_base', seed=config['seed'], config=config, label_maps=maps))
    save(root/'reference_template/plan.json', {'runs':baseline})
    control_cfg = dict(reference_analysis=str(root/'reference_template'), dataset_root=cfg['paths']['dataset'],
        splits=str(root/'splits/split_csv'), seed=cfg['seed'], training_control_repeats=3,
        training_batch_size=train['control_batch_size'], device=train['device'], workers=train['workers'],
        cache_enabled=False, wandb=train['wandb'])
    control_root = root/'controls'; control_root.mkdir()
    controls.prepare(control_cfg, control_root)
    control_plan = json.loads((control_root/'training/plan.json').read_text())
    for job in control_plan['jobs']:
        path = Path(job['config']); config = json.loads(path.read_text())
        config['training']['cpu_threads'] = train['cpu_threads']
        save(path, config); job['config_sha256'] = sha(path)
        jobs.append(dict(stage='matched_controls', index=job['id'], config=str(path),
            config_sha256=job['config_sha256'], output=config['output']['out_dir']))
    save(control_root/'training/plan.json', control_plan)
    analysis = copy.deepcopy(cfg)
    # New control plans belong to this run; do not import a release archive's plans.
    analysis['paths'].pop('control_source',None)
    analysis['paths'].update(trained_results=str(root/'reference'), splits=str(root/'splits/split_csv'),
        predictions=str(root/'inference'), features=str(root/'features'), analysis=str(root/'analysis'),
        adaptive_analysis=str(root/'adaptive_analysis'), controls=str(control_root),
        received_controls=str(control_root/'training/runs'),
        control_summary=str(control_root/'received_csv_summary/latest.json'))
    analysis_path=root/'analysis.yaml'; analysis_path.write_text(yaml.safe_dump(analysis, sort_keys=False))
    paper.load_settings(analysis_path)
    plan=dict(schema=1, identity=identity, jobs=jobs, analysis_config=str(analysis_path),
              counts={stage:sum(j['stage']==stage for j in jobs) for stage in (*STAGES,'matched_controls')})
    save(manifest, plan)
    return plan


def validate_plan(plan):
    for job in plan['jobs']:
        if sha(job['config']) != job['config_sha256']:
            raise ValueError(f'Frozen fit configuration changed: {job["config"]}')


def complete(job):
    receipt=Path(job['output'])/'training_complete.json'
    if not receipt.exists(): return False
    saved=json.loads(receipt.read_text())
    if saved['config_sha256']!=job['config_sha256']:
        raise ValueError('Completion receipt/config mismatch')
    run=Path(job['output'])/saved['result']['run_name']
    marker=Path(job['output'])/'run_status.txt'
    artifacts=all((run/f).is_file() for f in REQUIRED)
    if job['stage']=='matched_controls':
        artifacts=artifacts and (Path(job['output'])/'audit_complete.json').is_file()
    return artifacts and marker.is_file() and marker.read_text().strip()=='0'


def run_job(job):
    """One process/fit; partial or failed runs are never mistaken for completion."""
    import torch
    from worm_species.training import runner
    from worm_species.training.modes import resolve_configured_profile
    directory=Path(job['output']);directory.mkdir(parents=True, exist_ok=True)
    with lock(directory/'.training.lock'):
        if complete(job):
            print(f'Skipping completed {job["stage"]}:{job["index"]}', flush=True);return
        if list(directory.glob('*/config.json')):
            raise ValueError(f'Partial fit exists in {directory}; preserve it and use a fresh root (no silent resume)')
        config=json.loads(Path(job['config']).read_text())
        if sha(job['config'])!=job['config_sha256']:raise ValueError('Fit config changed')
        torch.set_num_threads(config['training']['cpu_threads'])
        try:
            if job['stage']=='matched_controls':
                result=controls.worker(job['config'])
            else:
                result=runner.run_one(config, replace(resolve_configured_profile(config),run_summary=True))
            run=directory/result['run_name']
            missing=[name for name in REQUIRED if not (run/name).is_file()]
            if missing:raise RuntimeError(f'Training did not produce required artifacts: {missing}')
            (directory/'run_status.txt').write_text('0\n')
            save(directory/'training_complete.json',dict(config_sha256=job['config_sha256'],result=result))
        except BaseException:
            (directory/'run_status.txt').write_text('1\n');raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan','run','status'])
    parser.add_argument('--config',default='configs/paper.yaml')
    parser.add_argument('--training-config',default='configs/train.yaml')
    parser.add_argument('--experiment',choices=[*STAGES,'matched_controls','all'],default='all')
    parser.add_argument('--index',type=int)
    parser.add_argument('--device')
    args=parser.parse_args()
    cfg, train, root=settings(args.config,args.training_config)
    if args.device:train['device']=args.device
    plan=prepare(cfg,train,root)
    jobs=[j for j in plan['jobs'] if args.experiment=='all' or j['stage']==args.experiment]
    if args.index is not None:
        if args.experiment=='all':parser.error('INDEX requires a named EXPERIMENT')
        jobs=[j for j in jobs if j['index']==args.index]
        if not jobs:parser.error('INDEX outside this experiment')
    print(json.dumps({'counts':plan['counts'],'analysis_config':plan['analysis_config']},indent=2),flush=True)
    if args.command=='run':
        for job in jobs:run_job(job)
    elif args.command=='status':
        print(json.dumps({'selected':len(jobs),'completed':sum(complete(j) for j in jobs)},indent=2))


if __name__=='__main__':main()
