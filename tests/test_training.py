"""Small CPU training-to-analysis contract check; no paper fit or download."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
import torch
import yaml

from scripts import paper_training as train
from scripts import compile_received_publication_controls as compiler
from worm_species.models.multitask import build_multitask_model

ROOT=Path(__file__).resolve().parents[1]


class TrainingTests(unittest.TestCase):
    def test_frozen_stage_counts_and_visual_conditions(self):
        from worm_species.config.normalization import normalize_config
        from worm_species.config.sweeps import generate_sweep_configs
        expected=dict(baseline=90,visual_ablation=840,visual_interactions=600,
                      adult_taxon_baseline=30,adult_taxon_holdouts=330)
        definitions=yaml.safe_load((ROOT/'configs/visual_conditions.yaml').read_text())
        for stage,count in expected.items():
            cfg=normalize_config(yaml.safe_load((ROOT/'configs/training'/f'{stage}.yaml').read_text()))
            fits=generate_sweep_configs(cfg)
            self.assertEqual(len(fits),count)
            self.assertNotIn('slurm',cfg)
            if stage in definitions:
                scientific=[{k:c[k] for k in ('name','transform','parameters')} for c in definitions[stage]]
                expanded=[{k:c['input_condition'][k] for k in ('name','transform','parameters')}
                          for c in fits[:len(scientific)]]
                self.assertEqual(expanded,scientific)

    def test_training_root_protects_reference_results(self):
        cfg=yaml.safe_load((ROOT/'configs/train.yaml').read_text())
        cfg['output_root']='outputs/training/new'
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'train.yaml';path.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError,'separate'):
                train.settings(ROOT/'configs/paper.yaml',path)

    def test_partial_runs_not_marked_complete(self):
        with tempfile.TemporaryDirectory() as d:
            job=dict(output=d,stage='baseline',index=0,config='unused',config_sha256='abc')
            self.assertFalse(train.complete(job))
            run=Path(d)/'partial';run.mkdir();(run/'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Partial fit'):
                train.run_job(job)
            self.assertFalse((Path(d)/'training_complete.json').exists())

    def test_cpu_checkpoint_predictions_and_control_scoring(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows={}
            for name,n in [('train',8),('val',4),('test',4)]:
                part=[]
                for i in range(n):
                    species='Alpha_one' if i%2==0 else 'Beta_two'
                    stage='Adult' if (i//2)%2==0 else 'Juvenile'
                    barcode=f'{species}_{stage}_{name}{i}'
                    for j in range(2):
                        filename=f'{barcode}_{j}.png'
                        Image.fromarray(np.random.default_rng(i+j).integers(0,255,(64,64,3),dtype=np.uint8)).save(root/filename)
                        part.append(dict(barcode=barcode,filename=filename,rel_path_seg=filename,
                            species_label=species,genus=species.split('_')[0],life_stage=stage))
                rows[name]=pd.DataFrame(part)
            reference=rows['train'].copy()
            rows['train']=reference.loc[~reference.barcode.eq(reference.barcode.iloc[0])].copy()
            splits=root/'split_csv';splits.mkdir()
            for name,frame in rows.items():frame.to_csv(splits/f'{name}_split.csv',index=False)
            pd.concat(rows.values()).to_csv(root/'metadata.csv',index=False)
            reference.to_csv(root/'full_train.csv',index=False)
            cfg=yaml.safe_load((ROOT/'configs/training/baseline.yaml').read_text())
            cfg['seed']=40;cfg['sweep']['enabled']=False
            cfg['data'].update(root_dir=str(root),metadata_csv=str(root/'metadata.csv'),image_size=64,
                min_individuals_per_class=1,min_individuals_per_class_by_task={'genus':1,'species':1,'age':1})
            cfg['split'].update(predefined_split_dir=str(root),use_predefined_splits=True)
            cfg['model'].update(name='resnet18',pretrained=False,freeze_backbone=True)
            cfg['training'].update(epochs=1,batch_size=4,num_workers=0,cpu_threads=2,device='cpu',val_interval=1,use_amp=False)
            cfg['cache']={'enabled':False};cfg['wandb']['enabled']=False
            cfg['output']['out_dir']=str(root/'run');cfg['early_stopping']['enabled']=False
            maps,_=train.build_label_maps(rows['train'],cfg['data']['target_cols'])
            cfg['audit_fixed_weight_reference']=str(root/'full_train.csv');cfg['audit_expected_label_maps']=maps
            config=root/'config.json';train.save(config,cfg)
            job=dict(stage='matched_controls',index=0,condition='full',seed=40,kind='reference',config=str(config),config_sha256=train.sha(config),output=cfg['output']['out_dir'])
            train.run_job(job)
            self.assertTrue(train.complete(job))
            from worm_species.training.losses import compute_individual_class_weights
            weights=json.loads((root/'run/fixed_class_weights.json').read_text())
            for task,column in cfg['data']['target_cols'].items():
                np.testing.assert_allclose(weights[task],compute_individual_class_weights(reference,column,'barcode',maps[task]))
            self.assertFalse(np.allclose(weights['species'],compute_individual_class_weights(rows['train'],'species_label','barcode',maps['species'])))
            receipt=json.loads((root/'run/training_complete.json').read_text())
            output=root/'run'/receipt['result']['run_name']
            checkpoint=torch.load(output/'best_model.pt',map_location='cpu',weights_only=False)
            model=build_multitask_model(cfg,{task:len(m) for task,m in maps.items()})
            model.load_state_dict(checkpoint['model_state'],strict=True);model.eval()
            with torch.inference_mode():
                prediction=model(torch.zeros(1,3,64,64))
            self.assertEqual({k:tuple(v.shape) for k,v in prediction.items()},
                             {k:(1,len(v)) for k,v in maps.items()})
            with patch.object(compiler,'TEST_SPLIT_PATH',str(splits/'test_split.csv')):
                scored=compiler.validate_predictions(job,cfg,root/'run')
            self.assertIsNotNone(scored)
            # A completed fit is reused, never trained a second time.
            train.run_job(job)


if __name__=='__main__':unittest.main()
