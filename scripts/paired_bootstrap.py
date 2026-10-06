"""Paired whole-worm uncertainty for saved matched-removal decisions; CPU only."""

from pathlib import Path

import hashlib,json,sys

import numpy as np

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]





sys.path.insert(0,str(ROOT/'src'))

from worm_species.deployment_audit import f1_fixed,matched_removal

def paired_interval(truth,predictions,target,strata,repeats=2000,seed=2026):
    """Predictions: stage-removal then three controls, by seed and worm.

    Share each worm resample across every model and condition. Stratify by
    recorded taxonomic category and life stage; condition on fitted models
    and the three fixed control subsets. No calibration/training refitting.
    """
    truth=np.asarray(truth,int);predictions=np.asarray(predictions,int)
    assert predictions.ndim==3 and predictions.shape[0]==4
    assert predictions.shape[2]==len(truth)==len(strata)
    rng=np.random.default_rng(seed);weights=np.zeros((repeats,len(truth)))
    for key in sorted(set(strata)):
        ids=np.flatnonzero(np.asarray(strata)==key)
        weights[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=repeats)
    assert np.all(weights.sum(axis=1)==len(truth))
    positive=truth==target;predicted=predictions==target
    def counts(mask):return (weights@mask.reshape(-1,len(truth)).T).reshape(repeats,*predictions.shape[:2])
    tp=counts(predicted&positive);fp=counts(predicted&~positive);fn=counts(~predicted&positive)
    denominator=2*tp+fp+fn
    f1=np.divide(2*tp,denominator,out=np.zeros_like(tp),where=denominator>0)
    draws=100*(f1[:,1:,:].mean(axis=1)-f1[:,0,:]).mean(axis=1)
    scores=np.array([[f1_fixed(truth,p,[target],max(int(truth.max()),int(predictions.max()))+1) for p in condition] for condition in predictions])
    mean=float(100*(scores[1:,:].mean(axis=0)-scores[0,:]).mean())
    # Independently reconstruct initial draws with the standard F1 scorer.
    for b in range(min(8,repeats)):
        index=np.repeat(np.arange(len(truth)),weights[b].astype(int))
        reconstructed=np.array([[f1_fixed(truth[index],p[index],[target],max(int(truth.max()),int(predictions.max()))+1) for p in condition] for condition in predictions])
        assert np.isclose(draws[b],100*(reconstructed[1:,:].mean(axis=0)-reconstructed[0,:]).mean(),atol=1e-10)
    return mean,np.quantile(draws,[.025,.975]),draws
