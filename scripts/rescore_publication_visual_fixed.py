"""Task-specific visual F1 from complete saved hard decisions, with worm voting."""
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import yaml
from worm_species.deployment_audit import f1_fixed


def rescore_visual(cfg,root):
    expected=json.loads((Path(cfg['reference_analysis'])/'plan.json').read_text())['runs'][0]['label_maps']
    rows=[];sources=[];counts=[];seen=set()
    # The gap-fill levels are already in the final visual config. A separate
    # recovery config is supplied for historical partial runs, not an extra fit.
    stages=['visual_ablation','visual_interactions','resolution_gapfill']
    required=set()
    for stage in stages[:2]:
        definitions=yaml.safe_load((ROOT/'configs/visual_conditions.yaml').read_text())[stage]
        seeds=json.loads((Path(cfg['reference_analysis'])/'plan.json').read_text())['runs']
        required|={(d['name'],r['seed']) for d in definitions for r in seeds if r['model']=='convnext_base'}
    for stage in stages:
        for p in sorted((Path(cfg['publication_results'])/'runs'/stage).glob('**/config.json')):
            c=json.loads(p.read_text());cond=c['input_condition'];key=(cond['name'],c['seed'])
            if key in seen:
                # Never select among two fits for the same scientific condition.
                raise ValueError(f'Duplicate visual condition/seed {key}; place gap-fill fits in missing slots only')
            if (p.parent.parent/'run_status.txt').read_text().strip()!='0':raise ValueError(f'Incomplete visual fit {p}')
            seen.add(key);maps=json.loads((p.parent/'label_to_index_by_task.json').read_text())
            if maps!=expected:raise ValueError('Visual output vocabulary differs from RGB reference')
            pred=p.parent/'test_predictions_best.csv'
            f=pd.read_csv(pred,usecols=['task','individual_id','true_label','predicted_label'],dtype={'individual_id':str})
            if f.groupby(['task','individual_id']).true_label.nunique().max()!=1:raise ValueError('Conflicting worm labels')
            votes=f.groupby(['task','individual_id','predicted_label']).size().rename('votes').reset_index()
            votes=votes.sort_values(['task','individual_id','votes','predicted_label'],ascending=[True,True,False,True]).drop_duplicates(['task','individual_id'])
            votes=votes.merge(f[['task','individual_id','true_label']].drop_duplicates(),on=['task','individual_id'],validate='one_to_one')
            transform=cond['transform'];params=cond.get('parameters',{});level=cond.get('strength',0)
            if transform=='resolution_loss':level=max(1,round(c['preprocessing']['image_size']*(1-params['percent']/100)))
            if transform=='patch_shuffle':level=params['grid_size']
            for task,g in votes.groupby('task'):
                truth=g.true_label.map(maps[task]);predicted=g.predicted_label.map(maps[task]);k=len(maps[task])
                if truth.isna().any() or predicted.isna().any():raise ValueError('Unknown label in visual predictions')
                rows.append(dict(condition=cond['name'],seed=c['seed'],series=stage,level=level,transform=transform,
                    parameters_json=json.dumps(params,sort_keys=True),task=task,
                    macro_f1=float(f1_fixed(truth,predicted,range(k),k))))
                counts.append(dict(condition=cond['name'],seed=c['seed'],task=task,individuals=len(g),images=len(f[f.task.eq(task)]),
                                   scored_classes=k,true_classes=g.true_label.nunique(),source=str(pred)))
            for q in [p,p.parent/'label_to_index_by_task.json',pred]:sources.append(dict(path=str(q),sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
    if missing:=required-seen:raise ValueError(f'Missing visual condition/seed pairs: {sorted(missing)[:10]} (total {len(missing)})')
    out=Path(root)/'tables';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(sources).drop_duplicates('path').to_csv(out/'input_visual_hashes.csv',index=False)
    pd.DataFrame(counts).to_csv(out/'visual_scoring_populations.csv',index=False)
    return pd.DataFrame(rows)
