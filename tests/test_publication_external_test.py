"""Exercise the external-test contract without Slurm, GPUs or trained weights."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

from scripts import run_publication_external_test as external


class TinyModel(torch.nn.Module):
    def forward(self, images):
        return {task: torch.tensor([[2., 0.]], device=images.device).repeat(len(images), 1)
                for task in external.TASKS}


class ExternalEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / 'dataset'
        self.results = self.root / 'old_results'
        self.output = self.root / 'external_results'
        self.maps = {'age': {'Adult': 0, 'Juvenile': 1},
                     'genus': {'Aporrectodea': 0, 'Lumbricus': 1},
                     'species': {'Aporrectodea_rosea': 0, 'Lumbricus_rubellus': 1}}
        self.cfg = {'model': {'name': 'convnext_base', 'pretrained': True}, 'seed': 40,
                    'preprocessing': {'image_size': 16}, 'training': {'use_amp': False},
                    'data': {'crop_to_foreground': False, 'barcode_col': 'barcode',
                             'target_cols': {'age': 'life_stage', 'genus': 'genus', 'species': 'species_label'},
                             'taxonomic_uncertainty': {'uncertain_species_labels': ['Lumbricus_sp']}}}
        self.rows = []
        for camera in external.CAMERAS:
            for index, taxon in enumerate(('Aporrectodea_rosea', 'Lumbricus_sp', 'Unknown_species')):
                relative = f'{camera}/{index}.jpg'
                path = self.dataset / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.arange(24*32*3, dtype=np.uint8).reshape(24, 32, 3)).save(path)
                self.rows.append({'year': '2026', 'camera': camera, 'kind': 'worm',
                                  'image_id': f'{camera}_{index}', 'individual_id': f'2026:{index}',
                                  'barcode': f'{taxon}_Adult_{index}', 'location_code': '04',
                                  'taxon': taxon, 'life_stage': 'Adult', 'original_paper_split': '',
                                  'segmentation_status': 'segmented', 'rel_path_seg': relative})
        external.save_csv(self.dataset / 'metadata/images.csv', self.rows)
        for model in external.MODELS:
            for seed in external.SEEDS:
                cfg = copy.deepcopy(self.cfg)
                cfg['model']['name'], cfg['seed'] = model, seed
                directory = self.results / 'runs/baseline' / f'{model}_{seed}' / 'fit'
                external.save_json(directory / 'config.json', cfg)
                external.save_json(directory / 'label_to_index_by_task.json', self.maps)
                (directory.parent / 'run_status.txt').write_text('0\n')
                torch.save({'cfg': cfg, 'label_to_index_by_task': self.maps,
                            'model_state': TinyModel().state_dict()}, directory / 'best_model.pt')
                external.save_csv(directory / 'test_predictions_best.csv', [
                    {'task': task, 'true_index': 0, 'predicted_index': 0} for task in external.TASKS])
        self.args = SimpleNamespace(output=self.output, dataset=self.dataset, publication_results=self.results,
                                    year=2026, batch_size=2, workers=0, mode='plan', account='worm-species',
                                    partition='gpu-short', cpus=8, memory='32G', time='02:00:00', max_active=12,
                                    report_cpus=4, report_memory='8G', report_time='00:30:00', report_partition='')
        self.quiet = redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def plan(self):
        return external.create_plan(self.args)

    def evaluate_first(self, plan):
        with patch('worm_species.models.multitask.build_multitask_model', return_value=TinyModel()) as build, \
                patch.dict('os.environ', {'SLURM_TMPDIR': str(self.root / 'scratch')}):
            external.worker(self.output, 0, 'cpu')
            self.assertFalse(build.call_args.args[0]['model']['pretrained'])

    def test_plan_has_all_ninety_checkpoints_and_is_immutable(self):
        plan = self.plan()
        self.assertEqual(len(plan['runs']), 90)
        self.assertEqual({r['seed'] for r in plan['runs']}, set(external.SEEDS))
        self.assertEqual(external.load_plan(self.output), plan)
        self.assertEqual(self.plan(), plan)
        self.rows[0]['location_code'] = '05'
        external.save_csv(self.dataset / 'metadata/images.csv', self.rows)
        with self.assertRaisesRegex(ValueError, 'another input snapshot'):
            self.plan()

    def test_discovery_requires_every_seed_and_successful_baselines(self):
        run = external.discover_checkpoints(self.results)[0]
        path = Path(run['checkpoint'])
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Missing checkpoint'):
            external.discover_checkpoints(self.results)
        shutil.rmtree(path.parent.parent)
        with self.assertRaisesRegex(ValueError, 'Need all 90'):
            external.discover_checkpoints(self.results)

    def test_cohorts_reject_pending_and_overlap_but_record_failures(self):
        self.rows[0]['segmentation_status'] = 'pending'
        external.save_csv(self.dataset / 'metadata/images.csv', self.rows)
        with self.assertRaisesRegex(ValueError, 'Segmentation is incomplete'):
            self.plan()
        self.rows[0]['segmentation_status'] = 'failed'
        external.save_csv(self.dataset / 'metadata/images.csv', self.rows)
        plan = self.plan()
        self.assertEqual(len(plan['excluded']), 1)
        self.assertEqual(len(plan['cohorts']['gphoto2']), 2)
        self.rows.append({**self.rows[0], 'year': '2025', 'camera': 'original_camera',
                          'original_paper_split': 'train', 'image_id': 'historical'})
        external.save_csv(self.dataset / 'metadata/images.csv', self.rows)
        with self.assertRaisesRegex(ValueError, 'barcodes overlap'):
            external.camera_cohorts(self.dataset, 2026)

    def test_cohort_path_must_stay_inside_dataset(self):
        self.rows[0]['rel_path_seg'] = '../outside.jpg'
        external.save_csv(self.dataset / 'metadata/images.csv', self.rows)
        with self.assertRaisesRegex(ValueError, 'Invalid segmented image path'):
            self.plan()

    def test_worker_scores_known_tasks_preserves_unknown_labels_and_resumes(self):
        plan = self.plan()
        self.evaluate_first(plan)
        for camera in external.CAMERAS:
            directory = external.run_dir(self.output, plan['runs'][0], camera)
            self.assertTrue(external.complete(directory, plan))
            rows = external.read_csv(directory / 'predictions.csv')
            self.assertEqual(len(rows), 9)
            self.assertEqual({r['location_code'] for r in rows}, {'04'})
            reasons = {r['unscored_reason'] for r in rows if r['scored'] == '0'}
            self.assertEqual(reasons, {'missing_or_uncertain_label', 'outside_checkpoint_vocabulary'})
            metrics = json.loads((directory / 'metrics.json').read_text())
            self.assertEqual(metrics['age']['n'], 3)
            self.assertEqual(metrics['genus']['n'], 2)
            self.assertEqual(metrics['species']['n'], 1)
        with patch('torch.load', side_effect=AssertionError('completed task loaded a checkpoint')), \
                patch.object(external, 'cache_images', side_effect=AssertionError('completed task staged images')):
            external.worker(self.output, 0, 'cpu')
        directory = external.run_dir(self.output, plan['runs'][0], 'webcam')
        (directory / 'predictions.csv').write_text('damaged')
        self.assertFalse(external.complete(directory, plan))
        (directory / 'complete.json').write_text('{')
        self.assertFalse(external.complete(directory, plan))

    def test_cache_matches_direct_preprocessing_and_rejects_changed_image(self):
        from worm_species.data.transforms import build_split_transform
        plan = self.plan()
        with patch.dict('os.environ', {'SLURM_TMPDIR': str(self.root / 'scratch')}):
            cache = external.cache_images(plan)
            row = plan['cohorts']['gphoto2'][0]
            transform = build_split_transform(split='test', preprocessing=self.cfg['preprocessing'])
            with Image.open(self.dataset / row['rel_path_seg']) as raw, \
                    Image.open(cache / (row['image_id'] + '.png')) as cached:
                self.assertTrue(torch.equal(transform(raw), transform(cached)))
            (self.dataset / row['rel_path_seg']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'image changed'):
                external.cache_images(plan)

    def test_metrics_handle_absent_classes_and_empty_scoring(self):
        metrics, cm = external.metric_record([0, 0], [0, 1], ['a', 'b', 'c'])
        self.assertEqual(metrics['balanced_accuracy'], .5)
        self.assertEqual(metrics['uniform_chance'], 1/3)
        self.assertEqual(cm.shape, (3, 3))
        metrics, _ = external.metric_record([], [], ['a', 'b'])
        self.assertIsNone(metrics['balanced_accuracy'])
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            external.metric_record([-1], [0], ['a', 'b'])

    def test_rendered_jobs_have_one_gpu_and_correct_dependency_position(self):
        plan = self.plan()
        array, report = external.render_jobs(self.output, plan, self.args)
        self.assertIn('--gpus=1', array)
        self.assertIn('--array=0-89%12', array)
        self.assertFalse(any('gpu' in arg for arg in report))
        for script in (array[-1], report[-1]):
            subprocess.run(['bash', '-n', script], check=True)
        self.assertEqual(external.array_spec([0, 3, 4, 5, 10]), '0,3-5,10')
        buffer = io.StringIO()
        with redirect_stdout(buffer), patch('subprocess.run', side_effect=AssertionError('dry run submitted')):
            external.submit(self.args, plan)
        self.assertLess(buffer.getvalue().index('--dependency=afterok:'),
                        buffer.getvalue().rindex('report.sh'))

    def test_submission_resumes_pending_only_and_refuses_active_jobs(self):
        plan = self.plan()
        self.args.mode = 'submit'
        def is_done(directory, unused):
            return directory != external.run_dir(self.output, plan['runs'][2], 'webcam')
        with patch.object(external, 'complete', side_effect=is_done), \
                patch('subprocess.run', side_effect=[SimpleNamespace(stdout='123;cluster\n'),
                                                     SimpleNamespace(stdout='124\n')]) as run:
            external.submit(self.args, plan)
        self.assertIn('--array=2%12', run.call_args_list[0].args[0])
        report = run.call_args_list[1].args[0]
        self.assertEqual(report[-2], '--dependency=afterok:123')
        receipt = json.loads((self.output / 'submission.json').read_text())
        self.assertEqual(receipt['array_indices'], [2])
        with patch('subprocess.run', return_value=SimpleNamespace(stdout='123_2\n')) as run:
            with self.assertRaisesRegex(RuntimeError, 'still active'):
                external.submit(self.args, plan)
        self.assertEqual(run.call_count, 1)
        with patch.object(external, 'complete', return_value=True), \
                patch('subprocess.run', side_effect=[SimpleNamespace(stdout=''), SimpleNamespace(stdout='125')]) as run:
            external.submit(self.args, plan)
        self.assertFalse(any('--array=' in arg for arg in run.call_args.args[0]))
        self.assertEqual(json.loads((self.output / 'submission.json').read_text())['report_job_id'], '125')

    def test_report_submission_failure_keeps_array_receipt(self):
        plan = self.plan()
        self.args.mode = 'submit'
        with patch('subprocess.run', side_effect=[SimpleNamespace(stdout='123\n'),
                                                  subprocess.CalledProcessError(1, 'sbatch')]):
            with self.assertRaises(subprocess.CalledProcessError):
                external.submit(self.args, plan)
        receipt = json.loads((self.output / 'submission.json').read_text())
        self.assertEqual(receipt['array_job_id'], '123')
        self.assertNotIn('report_job_id', receipt)

    def test_report_requires_all_seeds_and_generates_real_figures(self):
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, 'evaluations incomplete'):
            external.collect(self.output)
        self.evaluate_first(plan)
        for run in plan['runs'][1:]:
            for camera in external.CAMERAS:
                shutil.copytree(external.run_dir(self.output, plan['runs'][0], camera),
                                external.run_dir(self.output, run, camera))
        external.collect(self.output)
        report = self.output / 'summary'
        per_seed = pd.read_csv(report / 'per_seed_metrics.csv')
        self.assertEqual(len(per_seed), 3*30*3*3)
        summary = pd.read_csv(report / 'seed_summary.csv')
        self.assertEqual(set(summary.seeds), {30})
        self.assertTrue((summary.uniform_chance == .5).all())
        self.assertEqual(len(pd.read_csv(report / 'new_vs_original.csv')), 3*2*3*3)
        self.assertEqual(len(list((report / 'confusion_matrices').glob('*.png'))), 9)
        self.assertTrue((self.output / 'report_complete.json').is_file())


if __name__ == '__main__':
    unittest.main()
