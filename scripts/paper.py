#!/usr/bin/env python3
"""Paper reproduction commands. Neural training is submitted explicitly to Slurm."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import numpy as np
import pandas as pd
import yaml
from scripts import run_publication_external_test as io
from scripts import run_publication_domain_analysis as inference
from scripts import run_publication_domain_transfer as features
from scripts import run_publication_deployment_audit as analysis
from worm_species.domain_transfer import make_splits, make_trials

PROTOCOLS = {'global', 'within_species', 'across_species', 'leave_species_out', 'calibration_curve'}


def load_settings(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['external_inputs'] != ['rgb'] or cfg['direction'] != 'webcam_to_gphoto2':
        raise ValueError('This paper uses RGB external evaluation and USB-to-Canon correction only')
    if cfg['seeds'] != list(io.SEEDS) or cfg['models'] != list(io.MODELS):
        raise ValueError('The publication requires the frozen three architectures and 30 seeds')
    if cfg['methods'] != list(analysis.METHODS):
        raise ValueError('The paper comparison requires the six declared calibration methods')
    if cfg['folds'] != 5 or cfg['alignment_rank'] != 8:
        raise ValueError('The primary published design requires five folds and fixed rank eight')
    if cfg['common_species_indices'] != analysis.COMMON:
        raise ValueError('Common species must match the fixed output vocabulary of this paper')
    if any(cfg[k] < 1 for k in ['bootstrap_repeats', 'calibration_repeats', 'chance_repeats', 'view_repeats']):
        raise ValueError('Repetition counts must be positive')
    if cfg['training_control_repeats'] != 3:
        raise ValueError('The paired contrast requires three fixed matched-removal subsets')
    r=cfg['resources']
    if r['batch_size']<1 or r['workers']<0 or r['cpu_threads']<1:
        raise ValueError('Invalid inference batch size, workers or CPU threads')
    if any(n!='all' and (not isinstance(n,int) or n<5) for n in cfg['calibration_sizes']):
        raise ValueError('Fixed-rank calibration sizes must be at least five or all')
    if not 0<=cfg['projection_fold']<cfg['folds']:
        raise ValueError('Projection fold must be one of the frozen outer folds')
    if cfg['umap']['n_neighbors']<2 or not 0<=cfg['umap']['min_dist']<=1:
        raise ValueError('Invalid UMAP settings')
    for k, p in cfg['paths'].items():
        cfg['paths'][k] = str((ROOT / Path(p).expanduser()).resolve())
    inputs = [Path(cfg['paths'][k]) for k in ['dataset', 'original_data', 'splits', 'trained_results']]
    outputs = [Path(cfg['paths'][k]) for k in ['predictions', 'features', 'analysis', 'adaptive_analysis', 'controls']]
    for out in outputs:
        for src in inputs:
            if out == src or src in out.parents or out in src.parents:
                raise ValueError(f'Output must be separate from input: {out} / {src}')
    for i, a in enumerate(outputs):
        if any(a == b or a in b.parents or b in a.parents for b in outputs[i+1:]):
            raise ValueError('Independent stages must use separate output directories')
    return cfg


def inference_settings(cfg):
    return dict(dataset=cfg['paths']['dataset'], publication_results=cfg['paths']['trained_results'],
                output=cfg['paths']['predictions'], external_reference=None, year=2026,
                include_original_test=True, inputs=['rgb'], aggregation=['mean_probabilities'],
                representation=dict(model='convnext_base', selection_metric='loss', selection_mode='min',
                                    bootstrap_samples=cfg['bootstrap_repeats'], random_seed=cfg['seed'], projection='pca'),
                **cfg['resources'])


def analysis_settings(cfg, adaptive=False):
    return dict(reference_analysis=cfg['paths']['predictions'], reference_transfer=cfg['paths']['features'],
                dataset_root=cfg['paths']['dataset'], publication_results=cfg['paths']['trained_results'],
                splits=cfg['paths']['splits'], output=cfg['paths']['adaptive_analysis' if adaptive else 'analysis'],
                controls=cfg['paths']['controls'], training_run_root=cfg['paths']['received_controls'],
                seed=cfg['seed'], cpu_threads=cfg['resources']['cpu_threads'], bootstrap_repeats=cfg['bootstrap_repeats'],
                alignment_rank=cfg['adaptive_max_rank'] if adaptive else cfg['alignment_rank'], adaptive_rank=adaptive,
                ridge_alpha=cfg['ridge_alpha'], head_alpha=cfg['head_alpha'], calibration_sizes=cfg['calibration_sizes'],
                training_control_repeats=cfg['training_control_repeats'], training_batch_size=cfg['training_batch_size'],
                wandb=cfg['wandb'])


def transfer_plan(cfg, base):
    frame = features.metadata_frame(base)
    worms, views = make_splits(frame, cfg['folds'], cfg['seed'])
    trials = [t for t in make_trials(worms, views, cfg) if t['protocol'] in PROTOCOLS]
    if any(set(t['fit']) & set(t['test']) for t in trials):
        raise ValueError('Calibration/test worm overlap')
    selected, ranking = inference.choose_validation_seed(base['runs'],
                        {'model':'convnext_base', 'selection_metric':'loss', 'selection_mode':'min'})
    plan = dict(schema=1, runs=[dict(r, training_condition='rgb') for r in base['runs'] if r['model']=='convnext_base'],
                selected={'rgb': selected}, validation_ranking={'rgb': ranking}, metadata=frame.to_dict('records'),
                cohorts=base['cohorts'], dataset_root=base['dataset_root'], original_truth=base['original_truth'],
                reference_identity=base['identity'], worms=worms.to_dict('records'), views=views, trials=trials,
                direction='webcam_to_gphoto2',
                scientific_config={k:cfg[k] for k in ['seed','folds','calibration_sizes','calibration_repeats']},
                source_hashes={str(p.relative_to(ROOT)):io.sha(p) for p in [Path(__file__), ROOT/'src/worm_species/domain_transfer.py',ROOT/'scripts/run_publication_domain_transfer.py']})
    plan['identity'] = io.identity(plan)
    root = Path(cfg['paths']['features'])
    if (root/'plan.json').exists() and json.loads((root/'plan.json').read_text()) != plan:
        raise ValueError('Feature/calibration plan changed; choose a new features output root')
    io.save_json(root/'plan.json',plan)
    io.save_csv(root/'individual_splits.csv',plan['worms'])
    io.save_csv(root/'experiment_matrix.csv',[dict(t,fit=json.dumps(t['fit']),test=json.dumps(t['test'])) for t in trials])
    return plan


def freeze_analysis(cfg, root, stage):
    paths = [Path(cfg['reference_analysis'])/'plan.json', Path(cfg['reference_transfer'])/'plan.json',
             Path(cfg['dataset_root'])/'metadata/images.csv', Path(__file__),
             ROOT/'scripts/run_publication_deployment_audit.py', ROOT/'src/worm_species/deployment_audit.py',
             ROOT/'scripts/rescore_publication_visual_fixed.py']
    receipt = dict(config=cfg, sources={str(p):io.sha(p) for p in paths})
    receipt['identity'] = io.identity(receipt)
    if (root/'plan.json').exists() and json.loads((root/'plan.json').read_text()) != receipt:
        raise ValueError('Analysis identity changed; choose a new output root')
    io.save_json(root/'plan.json',receipt)
    return receipt['identity']


def run_analysis(cfg, stage, adaptive=False):
    from threadpoolctl import threadpool_limits
    opts=analysis_settings(cfg,adaptive);root=Path(opts['output']);root.mkdir(parents=True,exist_ok=True)
    (root/'tables').mkdir(exist_ok=True)
    identity=freeze_analysis(opts,root,stage)
    base=json.loads((Path(opts['reference_analysis'])/'plan.json').read_text())
    transfer=json.loads((Path(opts['reference_transfer'])/'plan.json').read_text())
    with io.lock(root/'.analysis.lock'), threadpool_limits(limits=opts['cpu_threads']):
        if stage in ['all','baseline']: analysis.baseline(opts,root,base)
        if stage in ['all','segmentation']: analysis.segmentation(opts,root,base,pd.read_csv(root/'baseline_predictions.csv.gz'))
        if stage in ['all','visual']: analysis.visual(opts,root)
        if stage in ['all','calibration']: analysis.corrections(opts,root,base,transfer,identity)
    io.save_json(root/f'{stage}_complete.json',dict(identity=identity,stage=stage))


def training_plan(cfg, submit=False):
    output=Path(cfg['paths']['trained_results']);runtime=output/'configuration';runtime.mkdir(parents=True,exist_ok=True)
    stages=[]
    for source in sorted((ROOT/'configs/training').glob('*.yaml')):
        if source.stem=='resolution_gapfill':continue  # Recovery subset, already included in visual_ablation.
        c=yaml.safe_load(source.read_text())
        c['data'].update(root_dir=cfg['paths']['original_data'],metadata_csv=str(Path(cfg['paths']['original_data'])/'01_Segmented/global_metadata.csv'))
        c['split']['predefined_split_dir']=str(Path(cfg['paths']['splits']).parent)
        # The historical loader looks for predefined_split_dir/split_csv. Supply
        # that layout explicitly rather than silently creating new random splits.
        if Path(cfg['paths']['splits']).name != 'split_csv':
            raise ValueError('Set paths.splits to a directory named split_csv containing the original partitions')
        c['output']['out_dir']=str(output/'runs'/source.stem)
        c['cache']['dir']=str(output/'image_cache')
        c['wandb'].update(mode=cfg['wandb']['mode'],project=cfg['wandb']['project'])
        target=runtime/source.name;target.write_text(yaml.safe_dump(c,sort_keys=False))
        stages.append({'name':source.stem,'config':str(target)})
    # Baseline must be first: it establishes the shared image cache.
    stages.sort(key=lambda s:(s['name']!='baseline',s['name']))
    cluster=yaml.safe_load((ROOT/'configs/genome.yaml').read_text())
    cluster['slurm']['paths'].update(project_root=str(ROOT),data_root=cfg['paths']['original_data'],
        metadata_csv=str(Path(cfg['paths']['original_data'])/'01_Segmented/global_metadata.csv'),
        results_root=str(output),cache_root=str(output/'image_cache'))
    cluster_path=runtime/'genome.yaml';cluster_path.write_text(yaml.safe_dump(cluster,sort_keys=False))
    pipeline=dict(name='paper-reproduction',cluster_config=str(cluster_path),paper_result_dir=str(output),
        dependency='afterok',required_hierarchy_loss_weights=[0.0],
        base_cache=dict(enabled=True,source_stage='baseline',directory_name='image_cache',cpus_per_task=8,memory='16G',time_limit='04:00:00'),
        condition_cache=dict(enabled=True,source_stages=['visual_ablation','visual_interactions'],
            consumer_stages=['visual_ablation','visual_interactions'],directory_name='condition_cache',
            transforms=['gaussian_blur_percent','patch_shuffle','resolution_loss','binary_mask','composed'],
            cpus_per_task=8,memory='64G',time_limit='04:00:00',max_active=12),stages=stages,report=dict(enabled=False))
    path=runtime/'pipeline.yaml';path.write_text(yaml.safe_dump(pipeline,sort_keys=False))
    argv=[sys.executable,str(ROOT/'scripts/run_ablation_pipeline.py'),'--pipeline',str(path),'--mode','submit' if submit else 'dry-run']
    subprocess.run(argv,cwd=ROOT,check=True)


def controls_slurm(cfg, submit=False):
    root=Path(cfg['paths']['controls']);plan=json.loads((root/'training/plan.json').read_text())
    cluster=yaml.safe_load((ROOT/'configs/genome.yaml').read_text())['slurm'];root.mkdir(parents=True,exist_ok=True)
    logs=root/'logs';logs.mkdir(exist_ok=True)
    job=root/'training/controls.sbatch'
    # Command operands are shell-quoted; resource directives are config values.
    for key in ['account','partition','time_limit','memory','cpus_per_task','gpus_per_task']:
        if '\n' in str(cluster[key]): raise ValueError('Invalid Slurm resource directive')
    text=f'''#!/bin/bash
#SBATCH --job-name=worm-paper-controls
#SBATCH --account={cluster['account']}
#SBATCH --partition={cluster['partition']}
#SBATCH --cpus-per-task={cluster['cpus_per_task']}
#SBATCH --mem={cluster['memory']}
#SBATCH --time={cluster['time_limit']}
#SBATCH --gres=gpu:{cluster['gpus_per_task']}
#SBATCH --array=0-{len(plan['jobs'])-1}%{cluster['array']['max_active']}
#SBATCH --output={logs}/%A_%a.log
set -euo pipefail
cd {shlex.quote(str(ROOT))}
'''
    # Expand HOME in a controlled Python path operation rather than quoting an
    # unexpanded shell placeholder. Environment name is always shell-quoted.
    conda_sh=os.path.expandvars(cluster['environment']['conda_sh'])
    text += 'source '+shlex.quote(conda_sh)+'\nconda activate '+shlex.quote(cluster['environment']['conda_env'])+'\n'
    text += f'export WORM_AUDIT_CPU_THREADS={int(cluster["cpus_per_task"])}\n'
    text += shlex.join(['python',str(ROOT/'scripts/paper.py'),'controls-worker','--config',str(cfg['_config_file'])])+' --index "$SLURM_ARRAY_TASK_ID"\n'
    job.write_text(text)
    print(f'Rendered {job}: {len(plan["jobs"])} fits; no training executed')
    if submit:
        if not subprocess.run(['which','sbatch'],capture_output=True).returncode==0:
            raise RuntimeError('Run training submission from the GenomeDK terminal')
        receipt=root/'training/submission.json'
        if receipt.exists(): raise ValueError('Submission receipt already exists; inspect jobs before resubmitting')
        result=subprocess.run(['sbatch','--parsable',str(job)],check=True,text=True,capture_output=True)
        io.save_json(receipt,dict(job_id=result.stdout.strip(),script_sha256=io.sha(job),plan_sha256=io.sha(root/'training/plan.json')))
        print(result.stdout.strip())


def compile_controls(cfg):
    from scripts import compile_received_publication_controls as compiler
    opts=analysis_settings(cfg);opts['output']=cfg['paths']['controls']
    plan=json.loads((Path(opts['output'])/'training/plan.json').read_text())
    test=Path(cfg['paths']['splits'])/'test_split.csv'
    if io.sha(test)!=plan['source_splits']['test']: raise ValueError('Local test split differs from frozen training partition')
    compiler.TEST_SPLIT_PATH=test
    runtime=Path(opts['output'])/'compile.yaml';runtime.write_text(yaml.safe_dump(opts,sort_keys=False))
    sys.argv=['compile_controls','--config',str(runtime),'--csv-root',cfg['paths']['received_controls']]
    compiler.main()


def verify(cfg):
    base=json.loads((Path(cfg['paths']['predictions'])/'plan.json').read_text())
    if len(base['runs']) != 90 or 'rgb' not in base['settings']['inputs']: raise ValueError('Invalid baseline inference plan')
    if any(r['label_maps']!=base['runs'][0]['label_maps'] for r in base['runs']): raise ValueError('Inconsistent output vocabulary')
    split_ids={}
    for split in ['train','val','test']:
        f=pd.read_csv(Path(cfg['paths']['splits'])/f'{split}_split.csv',dtype={'barcode':str})
        split_ids[split]=set(f.barcode)
    if any(split_ids[a]&split_ids[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Biological partition leakage')
    for key in ['analysis','adaptive_analysis']:
        path=Path(cfg['paths'][key])/'tables/calibration_summary.csv'
        if not path.is_file():raise FileNotFoundError(path)
    plan=json.loads((Path(cfg['paths']['features'])/'plan.json').read_text())
    if any(set(t['fit'])&set(t['test']) for t in plan['trials']): raise ValueError('Calibration/test leakage')
    if plan['direction']!='webcam_to_gphoto2':raise ValueError('Wrong correction direction')
    result=dict(inputs='RGB',direction=plan['direction'],biological_splits_disjoint=True,calibration_test_disjoint=True,
                runs=len(base['runs']),seeds=len(cfg['seeds']),folds=cfg['folds'])
    root=Path(cfg['paths']['analysis']);files=sorted((root/'tables').glob('*.csv'))
    io.save_json(root/'reproduction_receipt.json',dict(**result,table_sha256={p.name:io.sha(p) for p in files}))
    print(json.dumps(result,indent=2))


def main():
    commands=['check','plan','inference','features','analysis','adaptive','supplement','figures','verify','status',
              'dataset-plan','dataset-prepare','dataset-segment','training-plan','training-submit','controls-plan','controls-slurm-plan','controls-submit','controls-worker','controls-compile']
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=commands)
    parser.add_argument('--config',default=ROOT/'configs/paper.yaml',type=Path)
    parser.add_argument('--index',type=int,help='One checkpoint or controls-array index; omit to run all inference checkpoints')
    parser.add_argument('--stage',choices=['all','baseline','segmentation','visual','calibration'],default='all')
    parser.add_argument('--device',help='Runtime inference override, for example cpu or cuda:0')
    args=parser.parse_args();cfg=load_settings(args.config);cfg['_config_file']=str(args.config.resolve())
    if args.device:cfg['resources']['device']=args.device
    if args.command=='check':
        print('Configuration valid: RGB, USB-to-Canon, five folds, fixed rank 8, 30 seeds; no data/model execution.');return
    if args.command.startswith('dataset-'):
        acquisition=cfg['acquisition']
        argv=[sys.executable,str(ROOT/'scripts/build_publication_dataset.py'),'--mode',args.command.split('-',1)[1],
              '--old-root',cfg['paths']['original_data'],'--new-root',str((ROOT/acquisition['new_captures']).resolve()),
              '--split-root',cfg['paths']['splits'],'--output',cfg['paths']['dataset'],
              '--model',str((ROOT/acquisition['segmentation_checkpoint']).resolve()),'--device',cfg['resources']['device'],
              '--threads',str(cfg['resources']['cpu_threads'])]
        subprocess.run(argv,cwd=ROOT,check=True);return
    if args.command in ['training-plan','training-submit']:
        training_plan(cfg,args.command=='training-submit');return
    if args.command.startswith('controls-'):
        from scripts import run_publication_matched_controls as controls
        root=Path(cfg['paths']['controls']);root.mkdir(parents=True,exist_ok=True)
        opts=analysis_settings(cfg);opts['output']=str(root)
        if args.command=='controls-plan':controls.prepare(opts,root)
        elif args.command in ['controls-slurm-plan','controls-submit']:controls_slurm(cfg,args.command=='controls-submit')
        elif args.command=='controls-compile':compile_controls(cfg)
        else:
            if args.index is None:raise ValueError('controls-worker requires --index')
            plan=json.loads((root/'training/plan.json').read_text());controls.worker(plan['jobs'][args.index]['config'])
        return
    if args.command=='plan':
        base=inference.make_plan(inference_settings(cfg));transfer_plan(cfg,base);return
    if args.command in ['inference','features','status']:
        root=Path(cfg['paths']['predictions']);base=inference.load_plan(root)
        if args.command=='status':print(json.dumps(inference.status(root,base),indent=2));return
        if args.command=='inference':
            indices=range(len(base['runs'])) if args.index is None else [args.index]
            for index in indices:inference.worker(root,base,index,cfg['resources'])
        else:
            plan=json.loads((Path(cfg['paths']['features'])/'plan.json').read_text())
            plan=dict(plan,runs=[r for r in plan['runs'] if r.get('training_condition','rgb')=='rgb' and r['model']=='convnext_base'])
            if len(plan['runs'])!=30 or {r['seed'] for r in plan['runs']}!=set(cfg['seeds']):
                raise ValueError('Feature extraction requires exactly the 30 RGB ConvNeXt checkpoints')
            indices=range(len(plan['runs'])) if args.index is None else [args.index]
            for index in indices:
                if index<0 or index>=len(plan['runs']):raise ValueError('Feature checkpoint index outside plan')
                features.extract_features(Path(cfg['paths']['features']),plan,plan['runs'][index],cfg['resources'])
        return
    if args.command in ['analysis','adaptive']:
        run_analysis(cfg,'calibration' if args.command=='adaptive' else args.stage,args.command=='adaptive');return
    if args.command in ['supplement','figures']:
        from scripts import paper_supplement
        (paper_supplement.build if args.command=='supplement' else paper_supplement.figures)(cfg);return
    verify(cfg)


if __name__=='__main__':main()
