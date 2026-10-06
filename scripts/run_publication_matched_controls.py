"""Prepare/run new specimen-matched biological training controls, separately.

Training-only removals; unchanged validation/test; full-baseline class weights.
One GPU fit at a time. Existing experiments are never modified or resumed here.
"""

from __future__ import annotations

import argparse,copy,hashlib,json,os,sys,subprocess,fcntl,time

from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'scripts')]

import pandas as pd

import yaml

from worm_species.deployment_audit import matched_removal

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,default=str));temp.replace(path)

def execution_paths(config, mapping):
    """Relocate only filesystem fields; retain the original scientific config."""
    cfg=copy.deepcopy(config)
    fields=[('data','root_dir'),('data','metadata_csv'),('split','predefined_split_dir'),
            ('cache','dir'),('cache','root_dir_cache'),('output','out_dir'),
            ('audit_fixed_weight_reference',)]
    for field in fields:
        parent=cfg
        for key in field[:-1]:parent=parent.get(key,{})
        value=parent.get(field[-1])
        if not isinstance(value,str):continue
        for source,target in sorted(mapping.items(),key=lambda item:-len(item[0])):
            if value==source or value.startswith(source+'/'):
                parent[field[-1]]=target+value[len(source):];break
    return cfg

def prepare(cfg,root):
    if (root/'training/plan.json').exists():
        raise ValueError('A frozen control plan already exists; reuse it or choose a fresh output root')
    plan=json.loads((ROOT/cfg['reference_analysis']/'plan.json').read_text())
    template=next(r for r in plan['runs'] if r['model']=='convnext_base')['config']
    raw=pd.read_csv(Path(cfg['dataset_root'])/'metadata/images.csv')
    old=raw.loc[raw.year.eq(2025)&raw.kind.eq('worm')]
    if old.duplicated(['barcode','filename']).any():raise ValueError('Ambiguous historical metadata join')
    lookup=old.set_index(['barcode','filename'])
    splits={name:pd.read_csv(Path(cfg['splits'])/f'{name}_split.csv') for name in ['train','val','test']}
    for name,frame in splits.items():
        for column in ['rel_path_raw','rel_path_seg','rel_path_segmask']:
            frame[column]=[lookup.loc[(r.barcode,r.filename),column] for r in frame.itertuples()]
        if frame.rel_path_seg.isna().any():raise ValueError('Missing historical segmentation')
        for rel in frame.rel_path_seg:
            if not (Path(cfg['dataset_root'])/rel).is_file():raise FileNotFoundError(rel)
    ids={k:set(v.barcode) for k,v in splits.items()}
    if any(ids[a]&ids[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Original biological split leakage')
    directory=root/'training';directory.mkdir(exist_ok=True)
    full=directory/'full_train.csv';splits['train'].to_csv(full,index=False)
    metadata=directory/'metadata.csv';pd.concat(splits.values(),ignore_index=True).to_csv(metadata,index=False)
    conditions=[dict(name='full',kind='reference',species='',stage='',repeat=0,removed=[])]
    not_estimable=[]
    # Only resolved species with both stages support this stage-coverage contrast.
    for species,group in splits['train'].dropna(subset=['species_label']).groupby('species_label'):
        if group.life_stage.nunique()!=2:continue
        for stage in ['Adult','Juvenile']:
            target_name=f'{species}_{stage}'
            try:
                removed,_,_=matched_removal(splits['train'],species,stage,0,cfg['seed'])
            except ValueError as e:
                not_estimable.append(dict(species=species,stage=stage,reason=str(e)));continue
            conditions.append(dict(name=f'remove_{target_name}',kind='stage_removed',species=species,stage=stage,repeat=0,removed=removed))
            for repeat in range(cfg['training_control_repeats']):
                _,control,mismatch=matched_removal(splits['train'],species,stage,repeat,cfg['seed'])
                conditions.append(dict(name=f'random_{target_name}_repeat_{repeat}',kind='matched_random',species=species,stage=stage,repeat=repeat,removed=control,image_count_mismatch=mismatch))
    seeds=sorted(r['seed'] for r in plan['runs'] if r['model']=='convnext_base')
    jobs=[];counts=[]
    for condition in conditions:
        d=directory/'splits'/condition['name'];d.mkdir(parents=True,exist_ok=True)
        filtered=splits['train'].loc[~splits['train'].barcode.isin(condition['removed'])]
        if not set(filtered.dropna(subset=['species_label']).species_label)==set(splits['train'].dropna(subset=['species_label']).species_label):raise ValueError('A species output would disappear')
        for name in ['train','val','test']:
            (d/'split_csv').mkdir(exist_ok=True)
            (filtered if name=='train' else splits[name]).to_csv(d/'split_csv'/f'{name}_split.csv',index=False)
        counts.append(dict(**{k:v for k,v in condition.items() if k!='removed'},removed_individuals=len(condition['removed']),
            removed_images=len(splits['train'])-len(filtered),train_individuals=filtered.barcode.nunique(),train_images=len(filtered),
            validation_individuals=len(ids['val']),test_individuals=len(ids['test'])))
        for seed in seeds:
            c=copy.deepcopy(template);c['seed']=seed
            c['data'].update(root_dir=cfg['dataset_root'],metadata_csv=str(metadata))
            c['split'].update(predefined_split_dir=str(d),use_predefined_splits=True,save_splits=False)
            # Keep all new controls comparable on a 16 GB local GPU. This is a
            # new reference at batch 16, not a claim of identical legacy batches.
            c['training'].update(batch_size=cfg['training_batch_size'],num_workers=cfg.get('workers',2),device=cfg.get('device','auto'))
            c['cache']={'enabled':cfg.get('cache_enabled',False),'dir':str(directory/'image_cache'),'root_dir_cache':str(directory),
                        'format':'png','num_workers':2,'rebuild':False,'condition_variants':{'enabled':False}}
            c['sweep']={'enabled':False};c['input_condition']={'enabled':False};c['evaluation']={}
            c['data_holdout']={'enabled':False};c['test_cue_suppression']={'enabled':False}
            c['output']={'out_dir':str(Path(cfg.get('training_run_root',str(directory/'runs')))/condition['name']/f'seed_{seed}')}
            c['wandb'].update(project=cfg['wandb']['project'],mode=cfg['wandb']['mode'],group=condition['name'],tags=['deployment-audit-v2',condition['kind']])
            c['audit_fixed_weight_reference']=str(full);c['audit_expected_label_maps']=plan['runs'][0]['label_maps']
            path=directory/'configs'/condition['name']/f'seed_{seed}.json';save(path,c)
            jobs.append(dict(id=len(jobs),seed=seed,condition=condition['name'],kind=condition['kind'],config=str(path),config_sha256=digest(path)))
    # Run all conditions for a seed before moving to the next seed. The first
    # seed is selected by numeric ordering, never its test performance.
    jobs.sort(key=lambda j:(j['seed'],j['kind']!='reference',j['condition']))
    for i,j in enumerate(jobs):j['id']=i
    pd.DataFrame(counts).to_csv(directory/'conditions.csv',index=False)
    manifest=dict(schema=1,jobs=jobs,conditions=conditions,not_estimable=not_estimable,
        source_splits={k:digest(Path(cfg['splits'])/f'{k}_split.csv') for k in splits},
        design='Training-only whole-specimen removal; unchanged validation/test; fixed full-training weights and output vocabulary.',
        batch_size=cfg['training_batch_size'],reference_batch_size=template['training']['batch_size'],wandb=cfg['wandb'])
    path=directory/'plan.json'
    if path.exists() and json.loads(Path(path).read_text())!=manifest:raise ValueError('Training plan changed; choose a fresh output root')
    save(path,manifest);print(f'Prepared {len(jobs)} new full-network fits, {len(conditions)} conditions; none counted as run.',flush=True)

def worker(path):
    import torch
    from dataclasses import replace
    from worm_species.training import runner,losses
    from worm_species.training.modes import resolve_configured_profile
    cfg=json.loads(Path(path).read_text())
    requested=cfg['training'].get('device','auto')
    device=torch.device('cuda:0' if requested=='auto' and torch.cuda.is_available()
                        else 'cpu' if requested=='auto' else requested)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    cpu_threads=int(cfg['training'].get('cpu_threads',4))
    torch.set_num_threads(cpu_threads)
    save(Path(cfg['output']['out_dir'])/'runtime_resources.json',dict(cpu_threads=cpu_threads,
         loader_workers=cfg['training']['num_workers'],batch_size=cfg['training']['batch_size'],
         device=str(device),parallel_fits=1))
    reference=pd.read_csv(cfg['audit_fixed_weight_reference'])
    expected=cfg['audit_expected_label_maps']
    original=losses.build_criteria
    def fixed_criteria(train_df,target_cols,group_col,maps,device,**kwargs):
        if maps!=expected:raise ValueError('Output vocabulary changed in a matched control')
        return original(reference,target_cols,group_col,maps,device,**kwargs)
    runner.build_criteria=fixed_criteria
    profile=replace(resolve_configured_profile(cfg),run_summary=True)
    from worm_species.training.losses import compute_individual_class_weights
    weights={t:compute_individual_class_weights(reference,c,cfg['data']['group_col'],expected[t]).tolist() for t,c in cfg['data']['target_cols'].items()}
    save(Path(cfg['output']['out_dir'])/'fixed_class_weights.json',weights)
    source_files=[Path(__file__).resolve(),ROOT/'src/worm_species/training/runner.py',ROOT/'src/worm_species/training/loaders.py',ROOT/'src/worm_species/training/epochs.py',ROOT/'src/worm_species/deployment_audit.py']
    save(Path(cfg['output']['out_dir'])/'source_provenance.json',{str(p):digest(p) for p in source_files})
    try:
        result=runner.run_one(cfg,profile)
    finally:
        runner.build_criteria=original
    if device.type=='cuda':
        save(Path(cfg['output']['out_dir'])/'runtime_peak_gpu.json',dict(allocated_bytes=torch.cuda.max_memory_allocated(device),reserved_bytes=torch.cuda.max_memory_reserved(device)))
    save(Path(cfg['output']['out_dir'])/'audit_complete.json',dict(config_sha256=digest(path),result=result))
    return result
