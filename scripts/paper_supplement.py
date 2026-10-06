"""Additional paper tables from saved decisions; no neural-network execution."""
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import pandas as pd
from worm_species.deployment_audit import f1_fixed
from worm_species.domain_transfer import stable_seed
from scripts.paired_bootstrap import paired_interval
from scripts import run_publication_external_test as io


def save(root,name,frame):
    (root/'tables').mkdir(parents=True,exist_ok=True)
    frame.to_csv(root/'tables'/name,index=False)


def controls(cfg,root):
    source=json.loads(Path(cfg['paths']['control_summary']).read_text())
    if not source['complete_prediction_cohort']:raise ValueError('Matched training controls are incomplete; compile all planned fits first')
    path=Path(source['path']);plan=json.loads((Path(cfg['paths']['controls'])/'training/plan.json').read_text())
    if source['verified_prediction_fits']!=len(plan['jobs']):raise ValueError('Matched fit count differs from frozen plan')
    frame=pd.read_csv(path/'individual_predictions.csv.gz',dtype={'individual_id':str})
    meta=pd.read_csv(Path(cfg['paths']['splits'])/'test_split.csv',dtype={'barcode':str})
    worms=meta[['barcode','species_label','genus','life_stage']].drop_duplicates().set_index('barcode')
    if worms.index.duplicated().any():raise ValueError('Conflicting test-worm metadata')
    worms['category']=worms.species_label.fillna(worms.genus+' unresolved')
    maps=json.loads((Path(cfg['paths']['predictions'])/'plan.json').read_text())['runs'][0]['label_maps']
    frame=frame.merge(worms,left_on='individual_id',right_index=True,validate='many_to_one')
    seeds=cfg['seeds'];rows=[];boot=[];draws={}
    by={(name,task):g for (name,task),g in frame.groupby(['condition','task'])}
    for c in plan['conditions']:
        if c['kind']!='stage_removed':continue
        names=[c['name']]+[f"random_{c['species']}_{c['stage']}_repeat_{i}" for i in range(cfg['training_control_repeats'])]
        for task in ['genus','species']:
            groups=[by[(name,task)] for name in names];base=by[('full',task)]
            ids=sorted(groups[0].individual_id.unique());metadata=worms.loc[ids]
            if any(set(g.individual_id)!=set(ids) for g in groups+[base]):raise ValueError('Reference/removal test cohorts differ')
            truth=groups[0].pivot(index='seed',columns='individual_id',values='true_index').loc[seeds,ids].to_numpy()
            if not np.all(truth==truth[0]):raise ValueError('Truth differs across trained seeds')
            truth=truth[0]
            for g in groups+[base]:
                t=g.pivot(index='seed',columns='individual_id',values='true_index').loc[seeds,ids].to_numpy()
                if not np.array_equal(t,np.tile(truth,(len(seeds),1))):raise ValueError('Condition truth differs')
            pred=np.stack([g.pivot(index='seed',columns='individual_id',values='predicted_index').loc[seeds,ids].to_numpy() for g in groups])
            reference=base.pivot(index='seed',columns='individual_id',values='predicted_index').loc[seeds,ids].to_numpy()
            label=maps[task][c['species'] if task=='species' else c['species'].split('_')[0]]
            for cohort,stage in [('full_test',None),('omitted_stage',c['stage']),('retained_stage','Adult' if c['stage']=='Juvenile' else 'Juvenile')]:
                selected=np.ones(len(ids),bool) if stage is None else metadata.life_stage.eq(stage).to_numpy()
                if not selected.any() or not (truth[selected]==label).any():continue
                strata=(metadata.category+'|'+metadata.life_stage).to_numpy()[selected]
                mean,ci,sampling=paired_interval(truth[selected],pred[:,:,selected],label,strata,cfg['bootstrap_repeats'],cfg['seed'])
                excess=[]
                for i,seed in enumerate(seeds):
                    scores=[float(f1_fixed(truth[selected],p[i,selected],[label],len(maps[task]))) for p in pred]
                    ref=float(f1_fixed(truth[selected],reference[i,selected],[label],len(maps[task])))
                    delta=100*(np.mean(scores[1:])-scores[0]);excess.append(delta)
                    rows.append(dict(species=c['species'],stage_removed=c['stage'],task=task,cohort=cohort,seed=seed,
                        reference_class_f1=ref,removed_class_f1=scores[0],random_class_f1=np.mean(scores[1:]),excess_loss_pp=delta))
                boot.append(dict(species=c['species'],stage_removed=c['stage'],task=task,cohort=cohort,excess_mean=mean,
                    excess_sampling_ci_low=float(ci[0]),excess_sampling_ci_high=float(ci[1]),excess_seed_sd=np.std(excess,ddof=1),
                    individuals=int(selected.sum()),target_individuals=int((truth[selected]==label).sum()),seeds=len(seeds),control_subsets=3,
                    bootstrap_repeats=cfg['bootstrap_repeats'],conditional_on_fitted_models=True))
                draws['|'.join([c['species'],c['stage'],task,cohort])]=sampling
    save(root,'matched_stage_per_seed.csv',pd.DataFrame(rows))
    save(root,'matched_stage_bootstrap_summary.csv',pd.DataFrame(boot))
    save(root,'matched_bootstrap_draws.csv.gz',pd.DataFrame(draws))
    io.save_json(root/'matched_input_provenance.json',dict(plan_sha256=io.sha(Path(cfg['paths']['controls'])/'training/plan.json'),
                 prediction_sha256=io.sha(path/'individual_predictions.csv.gz'),test_split_sha256=io.sha(Path(cfg['paths']['splits'])/'test_split.csv')))


def chance_and_classes(cfg,root):
    baseline=pd.read_csv(root/'baseline_predictions.csv.gz');plan=json.loads((Path(cfg['paths']['predictions'])/'plan.json').read_text())
    maps=plan['runs'][0]['label_maps'];rows=[];common=cfg['common_species_indices']
    for (domain,task),g in baseline[baseline.model.eq('convnext_base')&baseline.seed.eq(cfg['seeds'][0])].groupby(['domain','task']):
        k=len(maps[task])
        for score in ['common_six','fixed_eight'] if task=='species' else ['full']:
            q=g[g.true_index.isin(common)] if score=='common_six' else g
            truth=q.true_index.to_numpy();labels=common if score=='common_six' else list(range(k))
            rng=np.random.default_rng(stable_seed(cfg['seed'],domain,task,score))
            values=np.array([f1_fixed(truth,rng.integers(k,size=len(truth)),labels,k) for _ in range(cfg['chance_repeats'])])
            rows.append(dict(domain=domain,task=task,score_set=score,individuals=len(truth),full_output_classes=k,
                        scored_classes=len(labels),chance_mean=values.mean(),chance_sd=values.std(ddof=1),random_prediction_repeats=len(values)))
    save(root,'chance_controls.csv',pd.DataFrame(rows));classes=[];confusions=[]
    for (model,seed,domain,task),g in baseline.groupby(['model','seed','domain','task']):
        k=len(maps[task]);names={index:name for name,index in maps[task].items()}
        for label in range(k):
            classes.append(dict(model=model,seed=seed,domain=domain,task=task,class_name=names[label],
                true_individuals=int(g.true_index.eq(label).sum()),class_f1=float(f1_fixed(g.true_index,g.predicted_index,[label],k))))
        for (truth,pred),h in g.groupby(['true_index','predicted_index']):
            confusions.append(dict(model=model,seed=seed,domain=domain,task=task,true_index=truth,predicted_index=pred,individuals=len(h)))
    per=pd.DataFrame(classes);save(root,'per_class_f1_per_seed.csv',per)
    save(root,'per_class_f1_summary.csv',per.groupby(['model','domain','task','class_name']).class_f1.agg(['mean','std','count']).reset_index().rename(columns={'std':'seed_sd','count':'seeds'}))
    save(root,'confusion_counts_per_seed.csv',pd.DataFrame(confusions))


def views(cfg,root):
    methods=['mean_probabilities','mean_logits','majority_vote'];rows=[];kinds={}
    for domain in ['original','gphoto2','webcam']:
        probe=pd.read_csv(Path(cfg['paths']['predictions'])/'predictions/convnext_base'/f"seed_{cfg['seeds'][0]}"/domain/'rgb/predictions.csv')
        kinds[domain]=int(probe[probe.task.eq('age')].groupby('individual_id').size().max())
    maximum=max(kinds.values());maps=json.loads((Path(cfg['paths']['predictions'])/'plan.json').read_text())['runs'][0]['label_maps']
    for domain in kinds:
        for seed in cfg['seeds']:
            path=Path(cfg['paths']['predictions'])/'predictions/convnext_base'/f'seed_{seed}'/domain/'rgb/predictions.csv'
            frame=pd.read_csv(path);frame=frame[frame.true_index.ge(0)]
            for task,data in frame.groupby('task'):
                groups=[]
                for worm,g in data.groupby('individual_id',sort=True):
                    g=g.sort_values('image_id')
                    if g.true_index.nunique()!=1:raise ValueError('Conflicting worm truth')
                    groups.append((worm,int(g.true_index.iloc[0]),np.array(g.probabilities_json.map(json.loads).tolist()),np.array(g.logits_json.map(json.loads).tolist())))
                truth=np.array([g[1] for g in groups]);k=len(maps[task])
                for repeat in range(cfg['view_repeats']):
                    predictions={m:np.empty((len(groups),maximum),int) for m in methods}
                    for i,(worm,_,p,l) in enumerate(groups):
                        # Permute available views once, then cycle if more views
                        # are requested. Reuse is explicit, not a new photograph.
                        rng=np.random.default_rng(cfg['seed']+repeat+int(hashlib.sha256(worm.encode()).hexdigest()[:8],16))
                        order=np.resize(rng.permutation(len(p)),maximum);p=p[order];l=l[order]
                        ps=p.cumsum(0);ls=l.cumsum(0);votes=np.eye(k,dtype=int)[p.argmax(1)].cumsum(0)
                        for n in range(maximum):
                            predictions['mean_probabilities'][i,n]=ps[n].argmax();predictions['mean_logits'][i,n]=ls[n].argmax()
                            tied=np.flatnonzero(votes[n]==votes[n].max());predictions['majority_vote'][i,n]=tied[np.argmax(ps[n,tied])]
                    for n in range(maximum):
                        for method in methods:
                            rows.append(dict(domain=domain,seed=seed,task=task,method=method,max_images=n+1,repeat=repeat,
                                biological_n=len(groups),mean_unique_images=np.mean([min(n+1,len(g[2])) for g in groups]),
                                worms_reusing_images=sum(len(g[2])<n+1 for g in groups),
                                macro_f1=float(f1_fixed(truth,predictions[method][:,n],range(k),k))))
    raw=pd.DataFrame(rows);save(root,'inference_views_repeats.csv',raw)
    keys=['domain','seed','task','method','max_images'];per=raw.groupby(keys).macro_f1.mean().reset_index();save(root,'inference_views_per_seed.csv',per)
    save(root,'inference_views_summary.csv',per.groupby(['domain','task','method','max_images']).macro_f1.agg(['mean','std','count']).reset_index().rename(columns={'std':'seed_sd','count':'seeds'}))
    io.save_json(root/'views_provenance.json',dict(maximum_images=maximum,domain_maximum=kinds,cycling_shorter_sets=True,
                                                score='individual macro-F1, fixed full output vocabulary'))


def dataset_counts(cfg,root):
    meta=pd.read_csv(Path(cfg['paths']['dataset'])/'metadata/images.csv',keep_default_na=False);rows=[]
    for (year,camera,kind,status),g in meta.groupby(['year','camera','kind','segmentation_status'],dropna=False):
        rows.append(dict(year=year,camera=camera,kind=kind,segmentation_status=status,images=len(g),individuals=g.individual_id.nunique()))
    save(root,'dataset_counts.csv',pd.DataFrame(rows))


def rank_comparison(cfg,root):
    rows=[]
    for name,strategy in [('analysis','fixed_8'),('adaptive_analysis','adaptive_up_to_32')]:
        f=pd.read_csv(Path(cfg['paths'][name])/'tables/calibration_per_seed_repeat.csv',dtype={'calibration_size':str})
        f=f[f.protocol.isin(['global','calibration_curve']) & f.method.isin(['none','mean_shift','coral','paired_ridge'])].copy()
        f['rank_strategy']=strategy;rows.append(f)
    per=pd.concat(rows,ignore_index=True)
    check=per[per.method.eq('none')&per.protocol.eq('global')].pivot(index=['seed','task'],columns='rank_strategy',values='f1')
    if not np.allclose(check.fixed_8,check.adaptive_up_to_32,atol=1e-10):raise ValueError('Rank sensitivity evaluation cohorts differ')
    save(root,'rank_comparison_per_seed_repeat.csv',per)
    keys=['protocol','calibration_size','method','task','rank_strategy']
    seeds=per.groupby(keys+['seed']).f1.mean().reset_index()
    save(root,'rank_comparison_summary.csv',seeds.groupby(keys).f1.agg(['mean','std','count']).reset_index().rename(columns={'std':'seed_sd','count':'seeds'}))


def projections(cfg,root):
    """One descriptive fit for all three domains; preserve coordinate provenance."""
    import umap
    from sklearn.decomposition import PCA
    plan=json.loads((Path(cfg['paths']['features'])/'plan.json').read_text());seed=plan['selected']['rgb']['seed'];features=[];records=[]
    for domain in ['original','gphoto2','webcam']:
        path=Path(cfg['paths']['features'])/'rgb'/f'seed_{seed}'/'features'/domain/'native'
        m=pd.read_csv(path/'metadata.csv',keep_default_na=False)
        with np.load(path/'features.npz',allow_pickle=False) as z:
            if list(z['image_id'])!=list(m.image_id):raise ValueError('Embedding identities differ')
            f=z['final'].astype(float)
        for worm,ids in m.groupby('individual_id',sort=True).indices.items():
            features.append(f[ids].mean(0));r=m.iloc[ids[0]]
            records.append(dict(domain=domain,individual_id=worm,genus=r.genus,species=r.species_label,stage=r.life_stage))
    x=np.stack(features);pca=PCA(n_components=2,random_state=cfg['seed']).fit_transform(x)
    reducer=umap.UMAP(**cfg['umap'],n_jobs=1)
    u=reducer.fit_transform(x);table=pd.DataFrame(records);table['pca_x']=pca[:,0];table['pca_y']=pca[:,1];table['umap_x']=u[:,0];table['umap_y']=u[:,1];table['seed']=seed
    save(root,'embedding_projection.csv',table)
    io.save_json(root/'projection_parameters.json',dict(selected_by='original validation loss only',selected_seed=seed,
        **cfg['umap'],normalisation='none',worm_mean_features=True,
        diagnostic='qualitative visualisation; no cluster-distance inference'))


def build(cfg):
    root=Path(cfg['paths']['analysis']);dataset_counts(cfg,root);chance_and_classes(cfg,root)
    controls(cfg,root);views(cfg,root);rank_comparison(cfg,root)
    from scripts.summarize_publication_calibration_donors import build_donor_summary
    from scripts.paper import analysis_settings
    build_donor_summary(root,analysis_settings(cfg));projections(cfg,root)
    representation_diagnostics(cfg,root);probability_projections(cfg,root);historical_biology(cfg,root)
    io.save_json(root/'supplement_complete.json',dict(no_neural_training=True,no_model_weights_loaded=True))


def figures(cfg):
    """Plot reproducible diagnostic figures from the exported CSVs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(cfg['paths']['analysis']);out=root/'figures';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def export(fig,name):
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=200,bbox_inches='tight');plt.close(fig)
    f=pd.read_csv(root/'tables/baseline_summary.csv');chance=pd.read_csv(root/'tables/chance_controls.csv')
    fig,axs=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,task in zip(axs,['age','genus','species']):
        q=f[f.task.eq(task)&f.cohort.eq('all_segmented')&(f.score_set.eq('common_six')|~f.task.eq('species'))]
        for model,g in q.groupby('model'):
            g=g.set_index('domain').loc[['original','gphoto2','webcam']]
            ax.errorbar(range(3),g['mean'],yerr=np.maximum(0,np.array([g['mean']-g.sampling_ci_low,g.sampling_ci_high-g['mean']])),fmt='o-',capsize=3,label=model)
        c=chance[chance.task.eq(task)&chance.score_set.eq('common_six' if task=='species' else 'full')].set_index('domain').loc[['original','gphoto2','webcam']]
        ax.plot(range(3),c.chance_mean,'k:',label='Uniform prediction chance')
        ax.set_xticks(range(3),['2025 Canon','2026 Canon','2026 USB'],rotation=15);ax.set_title(task.capitalize());ax.set_ylim(0,1)
    axs[0].set_ylabel('Individual macro-F1');axs[-1].legend(fontsize=9);export(fig,'external_domains')
    f=pd.read_csv(root/'tables/visual_task_summary.csv');fig,axs=plt.subplots(3,4,figsize=(14,10),sharey=True)
    for i,task in enumerate(['age','genus','species']):
        for j,transform in enumerate(['gaussian_blur_percent','resolution_loss','patch_shuffle','representations']):
            ax=axs[i,j];q=f[f.task.eq(task)&f['transform'].eq(transform)].sort_values('level')
            if transform=='representations':
                q=f[f.task.eq(task)&(f['transform'].isin(['binary_mask','saturation'])|f.condition.eq('resolution_loss_000pct'))]
                x=np.arange(len(q));ax.set_xticks(x,q.condition.str.replace('_',' '),rotation=30,ha='right')
            else:x=q.level
            ax.errorbar(x,q['mean'],yerr=q.seed_sd,fmt='o-',capsize=3)
            ref=f[f.task.eq(task)&f.condition.eq('resolution_loss_000pct')]['mean'].iloc[0]
            ax.axhline(ref,color='black',ls='--',label='Unperturbed reference')
            c=chance[chance.domain.eq('original')&chance.task.eq(task)&chance.score_set.eq('fixed_eight' if task=='species' else 'full')].chance_mean.iloc[0]
            ax.axhline(c,color='black',ls=':',label='Uniform prediction chance');ax.set_ylim(0,1)
            if i==0:ax.set_title(transform.replace('_',' ').capitalize())
            if j==0:ax.set_ylabel(task.capitalize()+' macro-F1')
    axs[0,0].legend(fontsize=8);export(fig,'visual_information')
    b=pd.read_csv(root/'tables/matched_stage_bootstrap_summary.csv');fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,task in zip(axs,['genus','species']):
        q=b[b.task.eq(task)&b.cohort.eq('full_test')]
        ax.errorbar(range(len(q)),q.excess_mean,yerr=np.maximum(0,[q.excess_mean-q.excess_sampling_ci_low,q.excess_sampling_ci_high-q.excess_mean]),fmt='o',capsize=4)
        ax.axhline(0,color='black');ax.set_xticks(range(len(q)),[r.species.replace('_',' ')+'\n'+r.stage_removed for r in q.itertuples()],rotation=30,ha='right');ax.set_title(task.capitalize()+' class F1')
    axs[0].set_ylabel('Excess loss versus matched removal (percentage points)');export(fig,'biological_coverage')
    f=pd.read_csv(root/'tables/calibration_summary.csv',dtype={'calibration_size':str});fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,task in zip(axs,['age','genus','species']):
        q=f[f.task.eq(task)&f.protocol.eq('global')];ax.bar(q.method,q.gain_pp);ax.axhline(0,color='black');ax.set_xticks(range(len(q)),q.method,rotation=40,ha='right');ax.set_title(task.capitalize())
    axs[0].set_ylabel('Gain in individual macro-F1 (percentage points)');export(fig,'calibration_gain')
    f=pd.read_csv(root/'tables/rank_comparison_summary.csv',dtype={'calibration_size':str});fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,task in zip(axs,['age','genus','species']):
        for (method,strategy),g in f[f.task.eq(task)&f.protocol.eq('calibration_curve')&f.method.isin(['coral','paired_ridge'])&~f.calibration_size.eq('all')].groupby(['method','rank_strategy']):
            g=g.assign(n=g.calibration_size.astype(int)).sort_values('n');ax.errorbar(g.n,g['mean'],yerr=g.seed_sd,fmt='o-',label=method+' '+strategy,capsize=3)
        ax.set_title(task.capitalize());ax.set_xlabel('Calibration worms per fold')
    axs[0].set_ylabel('Individual macro-F1');axs[-1].legend(fontsize=8);export(fig,'calibration_size_rank')
    p=pd.read_csv(root/'tables/embedding_projection.csv');fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,label in zip(axs,['genus','species']):
        categories=sorted(p[label].fillna('Unresolved').unique());colors={c:plt.get_cmap('tab10')(i%10) for i,c in enumerate(categories)}
        for domain,marker in [('original','o'),('gphoto2','s'),('webcam','^')]:
            q=p[p.domain.eq(domain)].fillna('Unresolved');ax.scatter(q.umap_x,q.umap_y,c=[colors[c] for c in q[label]],marker=marker,s=20,alpha=.6,label=domain)
        for _,g in p[p.domain.isin(['gphoto2','webcam'])].groupby('individual_id'):
            if len(g)==2:ax.plot(g.umap_x,g.umap_y,color='grey',alpha=.15,lw=.5)
        ax.set_title('UMAP coloured by '+label);ax.set_xlabel('UMAP 1');ax.set_ylabel('UMAP 2');ax.legend()
    export(fig,'domain_umap')
    p=pd.read_csv(root/'tables/fixed_rank_probability_umap_coordinates.csv');fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,task in zip(axs,['genus','species']):
        q=p[p.task.eq(task)];classes=sorted(q.true_class.unique());colors={label:plt.get_cmap('tab10')(i%10) for i,label in enumerate(classes)}
        for state,marker in [('USB before','s'),('USB ridge','o'),('Canon','^')]:
            t=q[q.state.eq(state)];ax.scatter(t.umap_1,t.umap_2,c=[colors[label] for label in t.true_class],marker=marker,s=25,alpha=.7,label=state)
        for _,g in q.groupby('individual_id'):
            g=g.set_index('state');a=g.loc['USB before',['umap_1','umap_2']].to_numpy(dtype=float);b=g.loc['USB ridge',['umap_1','umap_2']].to_numpy(dtype=float)
            ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='grey',alpha=.35,lw=.6))
        ax.set_title(task.capitalize()+' probabilities; held-out fold '+str(cfg['projection_fold']))
        ax.set_xlabel('UMAP 1 (arbitrary units)');ax.set_ylabel('UMAP 2 (arbitrary units)');ax.legend(fontsize=9)
    export(fig,'calibrated_probability_umap')
    io.save_json(out/'figure_parameters.json',dict(font_points=11,error_bars={'external':'worm bootstrap 95% CI','visual':'seed SD','biological':'paired worm bootstrap 95% CI','calibration_size':'seed SD'}))


def representation_diagnostics(cfg,root):
    """Paired distances at early/intermediate/final layers, held biological identity."""
    from worm_species.domain_analysis import paired_representation, bootstrap_mean
    plan=json.loads((Path(cfg['paths']['features'])/'plan.json').read_text());seed=plan['selected']['rgb']['seed']
    frames=[];arrays={}
    for domain in ['gphoto2','webcam']:
        path=Path(cfg['paths']['features'])/'rgb'/f'seed_{seed}'/'features'/domain/'native'
        meta=pd.read_csv(path/'metadata.csv',keep_default_na=False)
        with np.load(path/'features.npz',allow_pickle=False) as z:
            if list(z['image_id'])!=list(meta.image_id):raise ValueError('Embedding identities differ')
            for layer in ['early','intermediate','final','probabilities_age','probabilities_genus','probabilities_species']:
                arrays.setdefault(layer,[]).append(z[layer])
        frames.append(meta)
    meta=pd.concat(frames,ignore_index=True);summaries=[];per=[]
    for layer,values in arrays.items():
        table,_,_=paired_representation(meta,np.concatenate(values));table['layer']=layer;table['seed']=seed;per.append(table)
        for metric in ['paired_cosine_distance','paired_normalized_euclidean','different_individual_same_camera_cosine','camera_minus_different_individual_cosine']:
            summary=bootstrap_mean(table[metric],cfg['seed'],cfg['bootstrap_repeats']);summaries.append(dict(layer=layer,metric=metric,seed=seed,**summary))
    save(root,'representation_distances_per_individual.csv',pd.concat(per,ignore_index=True))
    save(root,'representation_distances_summary.csv',pd.DataFrame(summaries))


def probability_projections(cfg,root):
    """Reproduce the fixed-rank genus/species probability illustrations."""
    from umap import UMAP
    from scripts.run_publication_deployment_audit import load_features
    from worm_species.deployment_audit import fit_maps,corrected_logits,aggregate_probabilities
    plan=json.loads((Path(cfg['paths']['features'])/'plan.json').read_text());seed=plan['selected']['rgb']['seed']
    base=json.loads((Path(cfg['paths']['predictions'])/'plan.json').read_text());maps=base['runs'][0]['label_maps']
    source=load_features(Path(cfg['paths']['features']),seed,'webcam',maps);ref=load_features(Path(cfg['paths']['features']),seed,'gphoto2',maps)
    trial=next(t for t in plan['trials'] if t['protocol']=='global' and t['fold']==cfg['projection_fold'])
    ix=[source['lookup'][w] for w in trial['fit']];iy=[ref['lookup'][w] for w in trial['fit']]
    fitted=fit_maps(source['centroids'][ix],ref['centroids'][iy],cfg['alignment_rank'],cfg['ridge_alpha'],cfg['seed'])
    saved=pd.read_csv(root/'calibration'/f'seed_{seed}'/'predictions.csv.gz');saved=saved[saved.protocol.eq('global')&saved.fold.eq(cfg['projection_fold'])]
    rows=[]
    for task in ['genus','species']:
        states={}
        for name,pack,method in [('USB before',source,'none'),('USB ridge',source,'paired_ridge'),('Canon',ref,'none')]:
            logits=corrected_logits(pack['features'],*pack['weights'][task],fitted,method)
            pred,prob=aggregate_probabilities(logits,pack['indices'],len(pack['worms']))
            valid=[w for w in trial['test'] if pack['truths'][task][pack['lookup'][w]]>=0]
            positions=[pack['lookup'][w] for w in valid]
            if name!='Canon':
                expected=saved[saved.task.eq(task)&saved.method.eq(method)].set_index('individual_id').predicted_index
                if any(pred[p]!=expected.loc[w] for p,w in zip(positions,valid)):raise ValueError('UMAP illustration differs from frozen predictions')
            states[name]=(valid,prob[positions])
        if any(s[0]!=states['USB before'][0] for s in states.values()):raise ValueError('UMAP camera populations differ')
        normalise=lambda x:x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
        reducer=UMAP(**cfg['umap'],n_jobs=1)
        before=reducer.fit_transform(np.vstack([normalise(states['USB before'][1]),normalise(states['Canon'][1])]))
        n=len(states['USB before'][0]);coordinates={'USB before':before[:n],'Canon':before[n:],'USB ridge':reducer.transform(normalise(states['USB ridge'][1]))}
        names={v:k for k,v in maps[task].items()}
        for state,z in coordinates.items():
            for worm,point in zip(states[state][0],z):
                label=source['truths'][task][source['lookup'][worm]]
                rows.append(dict(task=task,individual_id=worm,true_class=names[label],state=state,umap_1=point[0],umap_2=point[1],seed=seed,fold=cfg['projection_fold'],rank=cfg['alignment_rank']))
    save(root,'fixed_rank_probability_umap_coordinates.csv',pd.DataFrame(rows))
    io.save_json(root/'probability_projection_parameters.json',dict(selected_seed=seed,selected_by='original validation loss',fold=cfg['projection_fold'],
        fit_states=['USB before','Canon'],transformed_state='USB ridge',normalisation='row L2',**cfg['umap'],fit_worms=len(trial['fit']),test_worms=len(trial['test']),visualisation_only=True))


def historical_biology(cfg,root):
    """Saved legacy exclusions, retaining unavailable-output status explicitly."""
    rows=[];provenance=[]
    for stage in ['adult_taxon_baseline','adult_taxon_holdouts']:
        for p in sorted((Path(cfg['paths']['trained_results'])/'runs'/stage).glob('**/config.json')):
            c=json.loads(p.read_text());maps=json.loads((p.parent/'label_to_index_by_task.json').read_text())
            if (p.parent.parent/'run_status.txt').read_text().strip()!='0':raise ValueError('Incomplete biological fit '+str(p))
            path=p.parent/'test_predictions_best.csv';f=pd.read_csv(path,dtype={'individual_id':str})
            hold=c.get('data_holdout',{});condition=hold.get('name','full');where=hold.get('where',{})
            for task,mapping in maps.items():
                g=f[f.task.eq(task)].copy();k=len(mapping)
                # Original legacy decisions preserve hard predictions; same
                # lexical majority-vote rule as original visual rescoring.
                v=g.groupby(['individual_id','predicted_label']).size().rename('votes').reset_index()
                v=v.sort_values(['individual_id','votes','predicted_label'],ascending=[True,False,True]).drop_duplicates('individual_id')
                v=v.merge(g[['individual_id','true_label']].drop_duplicates(),on='individual_id',validate='one_to_one')
                target=where.get(task);status='estimable';score=np.nan
                if target is not None and target not in mapping:
                    status='not_estimable: target is absent from the trained output vocabulary'
                elif len(v):
                    truth=v.true_label.map(mapping);pred=v.predicted_label.map(mapping)
                    if truth.isna().any() or pred.isna().any():raise ValueError('Unmapped historical biological label')
                    labels=[mapping[target]] if target is not None else list(range(k))
                    score=float(f1_fixed(truth,pred,labels,k))
                rows.append(dict(stage=stage,condition=condition,seed=c['seed'],task=task,target=target or '',
                    metric='class_f1' if target else 'macro_f1',f1=score,status=status,individuals=len(v),images=len(g),
                    removed_from_json=json.dumps(hold.get('remove_from',[])),original_output_vocabulary_json=json.dumps(mapping),
                    aggregation='majority_vote_lexical_ties'))
            provenance.append(dict(path=str(path),sha256=io.sha(path),config_sha256=io.sha(p)))
    save(root,'historical_biology_per_seed.csv',pd.DataFrame(rows));save(root,'input_historical_biology_hashes.csv',pd.DataFrame(provenance))
