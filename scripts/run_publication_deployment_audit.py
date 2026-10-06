"""Run a separate, resumable audit from frozen predictions and features."""

from __future__ import annotations

import argparse, hashlib, json, sys, time

from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT)]

import numpy as np

import pandas as pd

import yaml

from scipy.special import softmax

from threadpoolctl import threadpool_limits

from worm_species.deployment_audit import (METHODS, f1_fixed, sampling_summary, fit_maps,
    corrected_logits, apply_map, head_logits, aggregate_probabilities)

from worm_species.domain_transfer import distances, stable_seed

from run_publication_external_test import labeled_frame

TASKS=('age','genus','species')

DISPLAY={'original':'2025 Canon','gphoto2':'2026 Canon','webcam':'2026 USB'}

COMMON=[0,1,2,3,4,7]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    temp.replace(path)

def table(root,name,frame):
    (root/'tables').mkdir(exist_ok=True)
    frame.to_csv(root/'tables'/name,index=False)

def summarize(frame, keys, cfg, maps):
    rows=[]; boot={}
    for key,g in frame.groupby(keys,sort=True,dropna=False):
        key=key if isinstance(key,tuple) else (key,)
        desc=dict(zip(keys,key));task=desc['task']; k=len(maps[task])
        pivot=g.pivot(index='seed',columns='individual_id',values='predicted_index').sort_index().sort_index(axis=1)
        if pivot.isna().any().any(): raise ValueError('Different specimen sets across model seeds')
        truth=g.groupby('individual_id').true_index.agg(lambda x: int(x.iloc[0]) if x.nunique()==1 else -99).reindex(pivot.columns)
        if (truth<0).any(): raise ValueError('Conflicting/unknown labels in scored cohort')
        labels=COMMON if task=='species' and desc.get('score_set')=='common_six' else list(range(k))
        seed=stable_seed(cfg['seed'],desc.get('task'),desc.get('cohort'),desc.get('score_set'))
        info,b,_=sampling_summary(truth.to_numpy(),pivot.to_numpy(),labels,k,cfg['bootstrap_repeats'],seed)
        info.update(desc);info['images']=int(g.groupby('seed').n_images.sum().iloc[0]) if 'n_images' in g else np.nan
        rows.append(info);boot[key]=b
    return pd.DataFrame(rows),boot

def baseline(cfg,root,plan):
    rows=[];source=[];validation=[]
    for run in plan['runs']:
        model,seed=run['model'],run['seed']
        if run['label_maps']!=plan['runs'][0]['label_maps']:raise ValueError('Architecture output vocabularies differ')
        if run['config']['multi_task']['loss_weights']!=plan['runs'][0]['config']['multi_task']['loss_weights']:raise ValueError('Validation task weights differ')
        summary_path=Path(run['checkpoint']).parent/'run_summary.json'
        if not summary_path.exists():
            marker='/runs/baseline/'
            if marker not in str(summary_path):raise FileNotFoundError(summary_path)
            candidate=(Path(cfg['publication_results'])/'runs/baseline'/str(summary_path).split(marker,1)[1]).resolve()
            if not candidate.is_relative_to(Path(cfg['publication_results']).resolve()):raise ValueError('Unsafe relocated result path')
            summary_path=candidate
        s=json.load(open(summary_path))
        validation.append(dict(model=model,seed=seed,selection_metric=s['selection_metric'],best_val_score=s['best_val_score'],
                               best_epoch=s['best_epoch'],source=str(summary_path)))
        for domain in DISPLAY:
            path=ROOT/cfg['reference_analysis']/'predictions'/model/f'seed_{seed}'/domain/'rgb/predictions.csv'
            if not path.with_name('complete.json').exists(): raise ValueError(f'Missing prediction receipt {path}')
            receipt=json.loads(path.with_name('complete.json').read_text())
            if receipt['plan_identity']!=plan['identity'] or receipt['files']['predictions.csv']!=sha(path):
                raise ValueError('Prediction completion checksum/plan mismatch: '+str(path))
            f=pd.read_csv(path);source.append(dict(path=str(path),sha256=sha(path)))
            for (task,worm),group in f.loc[f.true_index.ge(0)].groupby(['task','individual_id']):
                if group.true_index.nunique()!=1: raise ValueError('Conflicting worm labels')
                probabilities=np.stack(group.probabilities_json.map(json.loads).to_list())
                rows.append(dict(model=model,seed=seed,domain=domain,task=task,individual_id=worm,
                    true_index=int(group.true_index.iloc[0]),predicted_index=int(probabilities.mean(axis=0).argmax()),n_images=len(group)))
        print(f'baseline {model} seed {seed}',flush=True)
    f=pd.DataFrame(rows);f.to_csv(root/'baseline_predictions.csv.gz',index=False)
    table(root,'input_prediction_hashes.csv',pd.DataFrame(source))
    v=pd.DataFrame(validation);table(root,'validation_architecture_per_seed.csv',v)
    if set(v.selection_metric)!= {'loss'}: raise ValueError('Architecture validation criteria differ')
    vs=v.groupby('model').agg(mean_validation_loss=('best_val_score','mean'),seed_sd=('best_val_score','std'),seeds=('seed','size')).reset_index()
    vs['validation_selected']=vs.mean_validation_loss.eq(vs.mean_validation_loss.min())
    table(root,'validation_architecture_summary.csv',vs)
    cohorts=[]
    for (model,seed,task),g in f.groupby(['model','seed','task']):
        paired=set(g.loc[g.domain.eq('gphoto2'),'individual_id']) & set(g.loc[g.domain.eq('webcam'),'individual_id'])
        for domain,h in g.groupby('domain'):
            q=h.copy();q['cohort']='all_segmented';q['score_set']='common_six' if task=='species' else 'full';cohorts.append(q.loc[q.true_index.isin(COMMON)] if task=='species' else q)
            if task=='species':
                q=h.copy();q['cohort']='all_segmented';q['score_set']='fixed_eight';cohorts.append(q)
            if domain!='original':
                q=h.loc[h.individual_id.isin(paired)].copy();q['cohort']='paired_segmented';q['score_set']='common_six' if task=='species' else 'full';cohorts.append(q)
    scored=pd.concat(cohorts,ignore_index=True)
    summary,boot=summarize(scored,['model','domain','task','cohort','score_set'],cfg,plan['runs'][0]['label_maps'])
    table(root,'baseline_summary.csv',summary)
    changes=[]
    for (model,task,cohort,score),g in summary.groupby(['model','task','cohort','score_set']):
        pairs=[('gphoto2','webcam')] if cohort=='paired_segmented' else [('original','gphoto2'),('original','webcam')]
        for a,b in pairs:
            if a not in set(g.domain) or b not in set(g.domain): continue
            x=g.loc[g.domain.eq(a)].iloc[0];y=g.loc[g.domain.eq(b)].iloc[0]
            # Independent reference/2026 specimens: independent bootstrap streams.
            bx=boot[(model,a,task,cohort,score)];by=boot[(model,b,task,cohort,score)]
            if a=='original': by=by[np.random.default_rng(stable_seed(model,task,a,b)).permutation(len(by))]
            delta=by-bx
            changes.append(dict(model=model,task=task,cohort=cohort,score_set=score,comparison=f'{b} minus {a}',
                difference_pp=100*(y['mean']-x['mean']),sampling_ci_low_pp=100*np.quantile(delta,.025),sampling_ci_high_pp=100*np.quantile(delta,.975)))
    table(root,'domain_differences.csv',pd.DataFrame(changes))
    return f

def segmentation(cfg,root,plan,baseline_frame):
    meta=pd.read_csv(Path(cfg['dataset_root'])/'metadata/images.csv',keep_default_na=False)
    raw=meta.loc[meta.year.astype(str).eq('2026') & meta.kind.eq('worm')].copy()
    raw=labeled_frame(raw.to_dict('records'),plan['runs'][0]['config'])
    summary=[]
    for (camera,taxon,stage),g in raw.groupby(['camera','taxon','life_stage']):
        counts=g.groupby('individual_id').segmentation_status.agg(lambda x:int(x.eq('segmented').sum()))
        summary.append(dict(domain=camera,taxon=taxon,life_stage=stage,raw_images=len(g),failed_images=int(g.segmentation_status.ne('segmented').sum()),
            individuals=len(counts),zero_usable_views=int(counts.eq(0).sum()),one_usable_view=int(counts.eq(1).sum()),multiple_usable_views=int(counts.gt(1).sum())))
    table(root,'segmentation_by_taxon_stage.csv',pd.DataFrame(summary))
    maps=plan['runs'][0]['label_maps'];rows=[]
    for task in TASKS:
        col={'age':'life_stage','genus':'genus','species':'species_label'}[task]
        for camera,g in raw.groupby('camera'):
            truth=g.groupby('individual_id')[col].first().map(maps[task]).dropna().astype(int)
            for (model,seed),q in baseline_frame.loc[baseline_frame.domain.eq(camera)&baseline_frame.task.eq(task)].groupby(['model','seed']):
                pred=q.set_index('individual_id').predicted_index.reindex(truth.index).fillna(-1).astype(int)
                image_n=q.set_index('individual_id').n_images.reindex(truth.index).fillna(0).astype(int)
                for worm in truth.index:
                    rows.append(dict(model=model,seed=seed,domain=camera,task=task,individual_id=worm,true_index=truth[worm],predicted_index=pred[worm],n_images=image_n[worm],cohort='raw_end_to_end',score_set='common_six' if task=='species' else 'full'))
    f=pd.DataFrame(rows);f=f.loc[~f.task.eq('species')|f.true_index.isin(COMMON)]
    f.to_csv(root/'end_to_end_predictions.csv.gz',index=False)
    s,_=summarize(f,['model','domain','task','cohort','score_set'],cfg,maps)
    coverage=f.groupby(['model','domain','task','seed']).predicted_index.agg(lambda x:x.ge(0).mean()).groupby(level=[0,1,2]).mean().rename('coverage').reset_index()
    s=s.merge(coverage,on=['model','domain','task']);table(root,'end_to_end_summary.csv',s)

def visual(cfg,root):
    from rescore_publication_visual_fixed import rescore_visual
    m=rescore_visual(cfg,root);table(root,'visual_task_per_seed.csv',m)
    s=m.groupby(['condition','series','level','transform','task'],dropna=False).macro_f1.agg(['mean','std','count']).reset_index().rename(columns={'std':'seed_sd','count':'seeds'})
    table(root,'visual_task_summary.csv',s)

def load_features(base,seed,camera,maps):
    path=base/'rgb'/f'seed_{seed}'/'features'/camera/'native'
    if not (path/'complete.json').exists(): raise ValueError(f'Missing features receipt {path}')
    m=pd.read_csv(path/'metadata.csv',keep_default_na=False)
    receipt=json.loads((path/'complete.json').read_text())
    if any(sha(path/name)!=digest for name,digest in receipt['files'].items()):
        raise ValueError('Feature receipt checksum mismatch')
    z=np.load(path/'features.npz',allow_pickle=False)
    if not np.array_equal(z['image_id'].astype(str),m.image_id.astype(str).to_numpy()): raise ValueError('Feature/metadata order mismatch')
    features=z['final'].astype(float)
    worms=sorted(m.individual_id.unique());lookup={w:i for i,w in enumerate(worms)};indices=m.individual_id.map(lookup).to_numpy()
    centroids=np.zeros((len(worms),features.shape[1]));np.add.at(centroids,indices,features)
    centroids/=np.bincount(indices)[:,None]
    truths={t:m.groupby('individual_id')[{'age':'life_stage','genus':'genus','species':'species_label'}[t]].first().reindex(worms).map(maps[t]).fillna(-1).astype(int).to_numpy() for t in TASKS}
    weights={t:(z[t+'_weight'].astype(float),z[t+'_bias'].astype(float)) for t in TASKS}
    return dict(meta=m,features=features,worms=worms,lookup=lookup,indices=indices,centroids=centroids,truths=truths,weights=weights)

def corrections(cfg,root,plan,transfer,identity):
    # Reuse the frozen biological folds and nested calibration samples. Only the
    # new fixed-rank methods are fitted; original transfer results remain intact.
    allowed={'global','within_species','across_species','leave_species_out','calibration_curve'}
    sizes={str(x) for x in cfg['calibration_sizes']}
    trials=[t for t in transfer['trials'] if t['protocol'] in allowed and
        (t['protocol']!='calibration_curve' or str(t['calibration_size']) in sizes)]
    for t in trials:
        if set(t['fit'])&set(t['test']): raise ValueError('Calibration/evaluation worm overlap')
    write_json(root/'calibration_trials.json',trials)
    maps=plan['runs'][0]['label_maps'];base=ROOT/cfg['reference_transfer']
    seeds=sorted(r['seed'] for r in plan['runs'] if r['model']=='convnext_base')
    all_sources=[]
    for seed in seeds:
        out=root/'calibration'/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():
            if json.load(open(out/'complete.json'))['identity']!=identity: raise ValueError('Stale completion receipt')
            receipt=json.loads((out/'complete.json').read_text())
            if not receipt.get('files') or any(sha(out/name)!=digest for name,digest in receipt['files'].items()):
                raise ValueError('Correction output checksum mismatch')
            for record in pd.read_csv(out/'feature_sources.csv').to_dict('records'):
                if sha(record['path'])!=record['sha256'] or sha(Path(record['path']).with_name('metadata.csv'))!=record['metadata_sha256']:
                    raise ValueError('Calibration input changed after completion')
            print(f'corrections seed {seed}: complete',flush=True);continue
        source=load_features(base,seed,'webcam',maps);ref=load_features(base,seed,'gphoto2',maps)
        for camera in ['webcam','gphoto2']:
            p=base/'rgb'/f'seed_{seed}'/'features'/camera/'native'
            all_sources.append(dict(seed=seed,domain=camera,path=str(p/'features.npz'),sha256=sha(p/'features.npz'),metadata_sha256=sha(p/'metadata.csv')))
        for task in TASKS:
            if not all(np.array_equal(a,b) for a,b in zip(source['weights'][task],ref['weights'][task])): raise ValueError('Camera classifier weights differ')
        results=[];alignment=[];failures=[];cache={}
        for trial in trials:
            key=tuple(trial['fit'])
            ix=np.array([source['lookup'][w] for w in key]);iy=np.array([ref['lookup'][w] for w in key])
            if key not in cache:
                rank=min(cfg['alignment_rank'],source['features'].shape[1],2*len(ix)-2) if cfg.get('adaptive_rank') else cfg['alignment_rank']
                if rank<1 or rank>min(source['features'].shape[1],2*len(ix)-2):
                    cache[key]=None
                else:
                    fitted=fit_maps(source['centroids'][ix],ref['centroids'][iy],rank,cfg['ridge_alpha'],cfg['seed'])
                    predictions={}
                    for method in METHODS:
                        for task in TASKS:
                            try:
                                logits=(head_logits(source['centroids'][ix],source['truths'][task][ix],source['features'],len(maps[task]),fitted,cfg['head_alpha']) if method=='head_ridge' else corrected_logits(source['features'],*source['weights'][task],fitted,method))
                                predicted,_=aggregate_probabilities(logits,source['indices'],len(source['worms']))
                                predictions[(method,task)]=predicted
                            except ValueError as e:
                                predictions[(method,task)]=None
                    cache[key]=(fitted,predictions)
            if cache[key] is None:
                failures.append(dict(trial_id=trial['id'],reason='not_estimable: fewer calibration worms than fixed rank requires'));continue
            fitted,pred=cache[key]
            test=np.array([source['lookup'][w] for w in trial['test']]);test_ref=np.array([ref['lookup'][w] for w in trial['test']])
            desc={k:trial[k] for k in ['id','protocol','fold','repeat','train_group','test_group','calibration_size']};desc['trial_id']=desc.pop('id')
            desc.update(seed=seed,fit_individuals=len(ix),test_individuals=len(test),rank=cache[key][0]['rank'] if cache[key] is not None else cfg['alignment_rank'],direction='webcam_to_gphoto2')
            for method in METHODS:
                if method!='head_ridge':
                    cosine,_=distances(apply_map(source['centroids'][test],fitted,method),ref['centroids'][test_ref])
                    for worm,d in zip(trial['test'],cosine): alignment.append(dict(**desc,method=method,individual_id=worm,cosine_distance=d))
                for task in TASKS:
                    if pred[(method,task)] is None:
                        failures.append(dict(trial_id=trial['id'],task=task,method=method,reason='not_estimable: no labelled calibration worms'));continue
                    for j,worm in zip(test,trial['test']):
                        truth=source['truths'][task][j]
                        if truth<0: continue
                        if task=='species' and truth not in COMMON: continue
                        results.append(dict(**desc,method=method,task=task,individual_id=worm,true_index=int(truth),predicted_index=int(pred[(method,task)][j])))
        pd.DataFrame(results).to_csv(out/'predictions.csv.gz',index=False)
        pd.DataFrame(alignment).to_csv(out/'alignment.csv.gz',index=False)
        write_json(out/'not_estimable.json',failures)
        pd.DataFrame(all_sources[-2:]).to_csv(out/'feature_sources.csv',index=False)
        write_json(out/'complete.json',dict(identity=identity,seed=seed,trials=len(trials),predictions=len(results),not_estimable=len(failures),files={name:sha(out/name) for name in ['predictions.csv.gz','alignment.csv.gz','not_estimable.json','feature_sources.csv']}))
        print(f'corrections seed {seed}: {len(trials)} trials, {len(cache)} distinct calibration fits',flush=True)
    table(root,'input_feature_hashes.csv',pd.concat([pd.read_csv(root/'calibration'/f'seed_{s}'/'feature_sources.csv') for s in seeds]))
    f=pd.concat([pd.read_csv(root/'calibration'/f'seed_{s}'/'predictions.csv.gz') for s in seeds],ignore_index=True)
    keys=['seed','protocol','repeat','calibration_size','train_group','test_group','method','task']
    rows=[]
    for key,g in f.groupby(keys):
        desc=dict(zip(keys,key));task=desc['task'];labels=COMMON if task=='species' else list(range(len(maps[task])))
        if desc['protocol'] not in {'global','calibration_curve'}:
            # Restricted-cohort diagnostics report class F1, not a changing macro average.
            labels=sorted(g.true_index.unique());desc['metric']='target_class_f1' if len(labels)==1 else 'cohort_macro_f1'
        else: desc['metric']='common_six_macro_f1' if task=='species' else 'macro_f1'
        desc.update(f1=float(f1_fixed(g.true_index,g.predicted_index,labels,len(maps[task]))),individuals=g.individual_id.nunique(),fit_min=g.fit_individuals.min(),fit_max=g.fit_individuals.max())
        rows.append(desc)
    per=pd.DataFrame(rows);table(root,'calibration_per_seed_repeat.csv',per)
    # Average calibration-subset repeats within each model seed before summarising seeds.
    aggkeys=['protocol','calibration_size','train_group','test_group','method','task','metric']
    seedmean=per.groupby(aggkeys+['seed'],dropna=False).f1.mean().reset_index()
    summary=seedmean.groupby(aggkeys,dropna=False).f1.agg(['mean','std','count']).reset_index().rename(columns={'std':'seed_sd','count':'seeds'})
    before=summary.loc[summary.method.eq('none'),aggkeys[:-3]+['task','metric','mean']].rename(columns={'mean':'baseline_mean'})
    # Explicit join keys avoid mixing species-specific and global scores.
    join=[k for k in aggkeys if k!='method']
    before=summary.loc[summary.method.eq('none'),join+['mean']].rename(columns={'mean':'baseline_mean'})
    summary=summary.merge(before,on=join);summary['gain_pp']=100*(summary['mean']-summary.baseline_mean)
    table(root,'calibration_summary.csv',summary)
    globalf=f.loc[f.protocol.eq('global')].copy();globalf['cohort']='paired_segmented';globalf['score_set']=np.where(globalf.task.eq('species'),'common_six','full')
    s,boots=summarize(globalf,['method','task','cohort','score_set'],cfg,maps)
    for i,row in s.iterrows():
        key=(row.method,row.task,row.cohort,row.score_set);basekey=('none',row.task,row.cohort,row.score_set)
        delta=boots[key]-boots[basekey]
        base_mean=float(s.loc[s.method.eq('none')&s.task.eq(row.task),'mean'].iloc[0])
        s.loc[i,'gain_pp']=100*(row['mean']-base_mean)
        s.loc[i,'gain_sampling_ci_low_pp']=100*np.quantile(delta,.025);s.loc[i,'gain_sampling_ci_high_pp']=100*np.quantile(delta,.975)
    table(root,'global_calibration_sampling_summary.csv',s)
    alignment=pd.concat([pd.read_csv(root/'calibration'/f'seed_{s}'/'alignment.csv.gz') for s in seeds],ignore_index=True)
    group=['protocol','calibration_size','train_group','test_group','method','seed']
    a=alignment.groupby(group).cosine_distance.mean().reset_index()
    table(root,'alignment_per_seed.csv',a)
    table(root,'alignment_summary.csv',a.groupby(group[:-1]).cosine_distance.agg(['mean','std','count']).reset_index())
