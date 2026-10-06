"""Portable guards and scientific split checks using small synthetic inputs."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import yaml
from scripts.paper import load_settings, PROTOCOLS
from scripts.paired_bootstrap import paired_interval
from scripts.run_publication_matched_controls import worker
from worm_species.domain_transfer import make_splits, make_trials

ROOT=Path(__file__).resolve().parents[1]


class ReproductionTests(unittest.TestCase):
    def config(self,edit=None):
        cfg=yaml.safe_load((ROOT/'configs/paper.yaml').read_text())
        if edit:edit(cfg)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'config.yaml';path.write_text(yaml.safe_dump(cfg))
            return load_settings(path)

    def test_camera_direction_and_input_guards(self):
        self.config()
        with self.assertRaisesRegex(ValueError,'RGB'):
            self.config(lambda c:c.update(external_inputs=['greyscale']))
        with self.assertRaisesRegex(ValueError,'USB-to-Canon'):
            self.config(lambda c:c.update(direction='unsupported'))

    def test_analysis_never_overwrites_inputs(self):
        with self.assertRaisesRegex(ValueError,'separate from input'):
            self.config(lambda c:c['paths'].update(analysis=c['paths']['trained_results']))
        with self.assertRaisesRegex(ValueError,'separate output'):
            self.config(lambda c:c['paths'].update(analysis=c['paths']['features']))

    def test_whole_worm_folds_and_nested_calibration(self):
        rows=[]
        for i in range(60):
            for camera in ['gphoto2','webcam']:
                for j in range(2):
                    rows.append(dict(domain=camera,image_id=f'{i}_{camera}_{j}',individual_id=f'{i:03}',
                        biological_group=f'species_{i%3}',group_kind='resolved_species',life_stage='Adult' if (i//3)%2 else 'Juvenile'))
        meta=pd.DataFrame(rows);worms,views=make_splits(meta,5,2026)
        self.assertEqual(len(worms),60);self.assertEqual(worms.fold.nunique(),5)
        for cameras in views.values():
            for split in cameras.values():self.assertFalse(set(split['fit'])&set(split['test']))
        cfg=dict(seed=2026,calibration_repeats=3,calibration_sizes=[5,10,20,'all'])
        trials=[t for t in make_trials(worms,views,cfg) if t['protocol'] in PROTOCOLS]
        for t in trials:self.assertFalse(set(t['fit'])&set(t['test']))
        global_tests=[set(t['test']) for t in trials if t['protocol']=='global']
        self.assertEqual(set().union(*global_tests),set(worms.individual_id))
        self.assertEqual(sum(map(len,global_tests)),60)
        for fold in range(5):
            selected=[t for t in trials if t['protocol']=='calibration_curve' and t['fold']==fold and t['repeat']==0]
            by_size={str(t['calibration_size']):set(t['fit']) for t in selected}
            self.assertLessEqual(by_size['5'],by_size['10']);self.assertLessEqual(by_size['10'],by_size['20']);self.assertLessEqual(by_size['20'],by_size['all'])
        bad=meta.copy();bad.loc[0,'biological_group']='conflicting_species'
        with self.assertRaisesRegex(ValueError,'Conflicting'):make_splits(bad,5,2026)

    def test_paired_bootstrap_preserves_perfect_contrast(self):
        truth=np.array([0,0,1,1]);wrong=1-truth
        p=np.stack([np.tile(wrong,(2,1))]+[np.tile(truth,(2,1))]*3)
        mean,ci,draws=paired_interval(truth,p,0,['A','A','B','B'],100,8)
        self.assertAlmostEqual(mean,100);np.testing.assert_array_equal(ci,[100,100]);np.testing.assert_array_equal(draws,np.full(100,100))
        p=np.stack([np.tile(truth,(2,1))]*4)
        mean,ci,_=paired_interval(truth,p,0,['A','A','B','B'],100,8)
        self.assertEqual(mean,0);np.testing.assert_array_equal(ci,[0,0])

    def test_training_worker_refuses_local_execution(self):
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'Slurm'):
                worker(Path('/nonexistent/config.json'))

    def test_configs_preserve_the_original_visual_intervention(self):
        cfg=yaml.safe_load((ROOT/'configs/training/visual_interactions.yaml').read_text())
        conditions=cfg['sweep']['conditions']
        self.assertTrue(conditions)
        for c in conditions:
            self.assertEqual(c['transform'],'composed')
            operations=c['parameters']['operations']
            self.assertEqual(operations[0]['transform'],'gaussian_blur_percent')
            self.assertIn(operations[1]['transform'],['patch_shuffle','saturation'])
        self.assertEqual(cfg['preprocessing']['image_size'],224)
        self.assertEqual(cfg['sweep']['parameters']['seed'],list(range(40,2941,100)))

    def test_all_command_has_no_neural_training_or_inference(self):
        text=(ROOT/'Makefile').read_text();recipe=text.split('\nall:\n',1)[1].split('\nplan:',1)[0]
        for term in ['training-submit','controls-submit','inference','features','dataset-segment']:
            self.assertNotIn(term,recipe)
        self.assertIn('controls-compile',recipe)


if __name__=='__main__':unittest.main()
