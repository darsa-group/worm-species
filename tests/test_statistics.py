import unittest
import numpy as np
import pandas as pd
from scipy.special import softmax
from worm_species.deployment_audit import (f1_fixed,sampling_summary,fit_maps,corrected_logits,apply_map,matched_removal,aggregate_probabilities,head_logits)

class DeploymentAuditTests(unittest.TestCase):
    def test_fixed_labels_keep_outside_prediction_errors(self):
        self.assertAlmostEqual(f1_fixed([0,1],[0,2],[0,1],3),.5)
        self.assertAlmostEqual(f1_fixed([0,1],[0,-1],[0,1],3),.5)
        self.assertAlmostEqual(f1_fixed([0,0,1,1],[0,1,1,1],[0,1],2),(2/3+4/5)/2)
    def test_bootstrap_repeats_specimens_and_preserves_models(self):
        a,b,s=sampling_summary([0,0,1,1],[[0,0,1,1],[0,0,1,1]],[0,1],2,100,9)
        self.assertEqual(a['individuals'],4);self.assertEqual(a['seeds'],2)
        np.testing.assert_array_equal(b,np.ones(100));self.assertEqual(a['seed_sd'],0)
        _,d,_=sampling_summary([0,0,1,1],[[1,1,0,0],[1,1,0,0]],[0,1],2,100,9)
        np.testing.assert_array_equal(b-d,np.ones(100))
    def test_mapping_recovers_translation_and_matches_feature_application(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(12,16));y=x+2
        fitted=fit_maps(x,y,8)
        w=rng.normal(size=(3,16));bias=rng.normal(size=3)
        for method in ['none','mean_shift','coral','paired_ridge','procrustes']:
            mapped=apply_map(x,fitted,method)
            np.testing.assert_allclose(corrected_logits(x,w,bias,fitted,method),mapped@w.T+bias,atol=1e-10)
        np.testing.assert_allclose(apply_map(x,fitted,'paired_ridge'),y,atol=1e-10)
        self.assertEqual(fitted['rank'],8)
        with self.assertRaisesRegex(ValueError,'fixed rank'):fit_maps(x[:4],y[:4],8)
    def test_average_probabilities_uses_all_views_and_abstains(self):
        pred,prob=aggregate_probabilities(np.array([[2,0],[0,3],[4,0]]),np.array([0,0,1]),3)
        np.testing.assert_allclose(prob[0],softmax([[2,0],[0,3]],axis=1).mean(axis=0))
        self.assertEqual(pred[-1],-1)
    def test_head_retains_full_output_vocabulary(self):
        x=np.random.default_rng(1).normal(size=(8,12));fit=fit_maps(x,x+1,8)
        logits=head_logits(x,np.array([0,0,1,1,0,1,-1,-1]),x,4,fit)
        self.assertEqual(logits.shape,(8,4))
    def test_matched_control_removes_whole_specimens_preserving_stages(self):
        rows=[dict(barcode=str(i),species_label='S',life_stage='Adult' if i<3 else 'Juvenile',filename=f'{i}_{j}') for i in range(8) for j in range(2)]
        f=pd.DataFrame(rows);target,control,mismatch=matched_removal(f,'S','Adult',1)
        self.assertEqual(len(target),len(control));self.assertEqual(mismatch,0)
        self.assertEqual(f.loc[~f.barcode.isin(control)].life_stage.nunique(),2)
        self.assertEqual((target,control,mismatch),matched_removal(f,'S','Adult',1))


if __name__=='__main__': unittest.main()
