#!/usr/bin/env python3
"""Pool all held-out taxa under each donor-fitted mapping for class F1.

A single-species test cohort cannot measure false positives from other taxa.
These tables pool all held-out target groups, with the same calibration mapping
within each outer fold, to preserve those errors. No additional fitting occurs.
"""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from worm_species.deployment_audit import f1_fixed


def build_donor_summary(root,cfg):
    root=Path(root);plan=json.loads((ROOT/cfg['reference_analysis']/'plan.json').read_text());maps=plan['runs'][0]['label_maps'];rows=[]
    from scripts.run_publication_external_test import labeled_frame
    paired=set(r['individual_id'] for r in plan['cohorts']['gphoto2']) & set(r['individual_id'] for r in plan['cohorts']['webcam'])
    meta=labeled_frame(plan['cohorts']['webcam'],plan['runs'][0]['config']).drop_duplicates('individual_id')
    meta=meta[meta.individual_id.isin(paired)]
    expected_counts={t:len(meta[meta[col].map(maps[t]).isin([0,1,2,3,4,7] if t=='species' else list(range(len(maps[t]))))]) for t,col in plan['runs'][0]['config']['data']['target_cols'].items()}
    cols=['seed','protocol','train_group','method','task','individual_id','true_index','predicted_index']
    for seed in range(40,2941,100):
        path=root/'calibration'/f'seed_{seed}'/'predictions.csv.gz'
        chunks=[g.loc[g.protocol.isin(['within_species','across_species'])] for g in pd.read_csv(path,usecols=cols,chunksize=100000)]
        f=pd.concat(chunks,ignore_index=True)
        for (donor,method,task),g in f.groupby(['train_group','method','task']):
            if g.individual_id.duplicated().any():raise ValueError('A donor/method predicts a test worm more than once')
            expected=expected_counts[task]
            if len(g)!=expected:raise ValueError('Incomplete mixed-taxon donor evaluation')
            labels=[0,1,2,3,4,7] if task=='species' else list(range(len(maps[task])))
            names={index:name for name,index in maps[task].items()}
            for label in labels:
                rows.append(dict(seed=seed,calibration_group=donor,method=method,task=task,test_class=names[label],
                    class_f1=float(f1_fixed(g.true_index,g.predicted_index,[label],len(maps[task]))),individuals=len(g),
                    true_class_individuals=int(g.true_index.eq(label).sum()),evaluation='all_heldout_biological_groups'))
    per=pd.DataFrame(rows);per.to_csv(root/'tables/calibration_donor_per_class_per_seed.csv',index=False)
    keys=['calibration_group','method','task','test_class','evaluation']
    summary=per.groupby(keys).class_f1.agg(['mean','std','count']).reset_index().rename(columns={'std':'seed_sd','count':'seeds'})
    join=[k for k in keys if k!='method'];baseline=summary.loc[summary.method.eq('none'),join+['mean']].rename(columns={'mean':'baseline_mean'})
    summary=summary.merge(baseline,on=join);summary['gain_pp']=100*(summary['mean']-summary.baseline_mean)
    summary.to_csv(root/'tables/calibration_donor_per_class_summary.csv',index=False)
    macro=per.groupby(['seed','calibration_group','method','task','evaluation']).class_f1.mean().reset_index(name='macro_f1')
    macro.to_csv(root/'tables/calibration_donor_macro_per_seed.csv',index=False)
    (root/'donor_summary_complete.json').write_text(json.dumps({'seeds':30,'evaluation':'Mixed held-out groups, one prediction per worm per donor/method, retaining false positives from other taxa.','additional_model_fits':0},indent=2)+'\n')

if __name__=='__main__':
    import argparse,yaml
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='dev/publication_deployment_audit_v2.yaml');a=ap.parse_args();cfg=yaml.safe_load(open(a.config));build_donor_summary(ROOT/cfg['output'],cfg)
