"""Independent deployment diagnostics; fixed vocabularies and specimen resampling.

No image encoder is trained here. Corrections map USB/webcam features to Canon/
gphoto2 features; all fitted quantities use calibration specimens only.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.special import softmax
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

METHODS = ('none', 'mean_shift', 'coral', 'paired_ridge', 'procrustes', 'head_ridge')


def f1_fixed(truth, predicted, labels, vocabulary):
    """Outside-set predictions remain errors; -1 predictions are abstentions."""
    truth, predicted = np.asarray(truth, int), np.asarray(predicted, int)
    if truth.shape != predicted.shape or np.any((truth < 0) | (truth >= vocabulary)):
        raise ValueError('Truth must have known labels in the full vocabulary')
    if np.any((predicted < -1) | (predicted >= vocabulary)):
        raise ValueError('Prediction outside the full vocabulary')
    cm = np.zeros((vocabulary, vocabulary + 1), float)
    np.add.at(cm, (truth, np.where(predicted < 0, vocabulary, predicted)), 1)
    return f1_counts(cm, labels)


def f1_counts(cm, labels):
    cm = np.asarray(cm, float)
    k = cm.shape[-2]
    tp = np.diagonal(cm[..., :k], axis1=-2, axis2=-1)
    denominator = cm.sum(axis=-1) + cm[..., :k].sum(axis=-2)
    values = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return values[..., list(labels)].mean(axis=-1)


def sampling_summary(truth, predictions, labels, vocabulary, repeats=2000, seed=2026):
    """Cross seeds and class-stratified whole-specimen bootstrap resampling.

    predictions is (models, specimens). Models are not biological replicates.
    Sampling CI holds all models fixed; crossed CI also resamples model seeds.
    Fitted maps remain fixed: these intervals are conditional on calibration.
    """
    truth, predictions = np.asarray(truth, int), np.asarray(predictions, int)
    if predictions.ndim != 2 or predictions.shape[1] != len(truth) or not len(truth):
        raise ValueError('Expected non-empty model-by-specimen predictions')
    scores = np.array([f1_fixed(truth, p, labels, vocabulary) for p in predictions])
    rng = np.random.default_rng(seed)
    weights = np.zeros((repeats, len(truth)), dtype=np.float64)
    for label in np.unique(truth):
        ids = np.flatnonzero(truth == label)
        weights[:, ids] = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=repeats)
    per_model = []
    for predicted in predictions:
        contributions = np.zeros((len(truth), vocabulary, vocabulary + 1))
        contributions[np.arange(len(truth)), truth, np.where(predicted < 0, vocabulary, predicted)] = 1
        cm = (weights @ contributions.reshape(len(truth), -1)).reshape(repeats, vocabulary, vocabulary + 1)
        per_model.append(f1_counts(cm, labels))
    per_model = np.array(per_model).T
    sampling = per_model.mean(axis=1)
    model_weights = rng.multinomial(len(scores), np.full(len(scores), 1 / len(scores)), size=repeats) / len(scores)
    crossed = (per_model * model_weights).sum(axis=1)
    return dict(mean=float(scores.mean()), seed_sd=float(scores.std(ddof=1)) if len(scores)>1 else 0.,
                sampling_ci_low=float(np.quantile(sampling,.025)), sampling_ci_high=float(np.quantile(sampling,.975)),
                crossed_ci_low=float(np.quantile(crossed,.025)), crossed_ci_high=float(np.quantile(crossed,.975)),
                individuals=len(truth), seeds=len(scores)), sampling, scores


def fit_maps(source, reference, rank=8, alpha=.1, seed=2026):
    source, reference = np.asarray(source, float), np.asarray(reference, float)
    if source.shape != reference.shape or source.ndim != 2 or not np.isfinite(source).all() or not np.isfinite(reference).all():
        raise ValueError('Invalid calibration centroids')
    if rank > min(source.shape[1], 2 * len(source) - 2):
        raise ValueError('not_estimable: requested fixed rank exceeds calibration capacity')
    a, b = source.mean(axis=0), reference.mean(axis=0)
    basis = PCA(n_components=rank, svd_solver='randomized', random_state=seed).fit(np.r_[source-a, reference-b]).components_.T
    x, y = (source-a) @ basis, (reference-b) @ basis
    scale = max(float(np.sqrt(np.mean(x*x))), 1e-8)
    x, y = x/scale, y/scale
    eye = np.eye(rank)
    ridge = eye + np.linalg.solve(x.T@x + alpha*len(x)*eye, x.T@(y-x))
    def power(c, exponent):
        val, vec = np.linalg.eigh(c)
        floor = max(float(np.trace(c)/rank)*1e-6, 1e-10)
        return (vec*np.maximum(val,floor)**exponent) @ vec.T
    coral = power(LedoitWolf().fit(x).covariance_, -.5) @ power(LedoitWolf().fit(y).covariance_, .5)
    rotation, _ = orthogonal_procrustes(x,y)
    return dict(source_mean=a, target_mean=b, basis=basis, scale=scale,
                matrices={'coral':coral,'paired_ridge':ridge,'procrustes':rotation}, rank=rank)


def corrected_logits(features, weight, bias, fitted, method):
    """Algebraically apply the mapping through a frozen linear classifier."""
    features, weight = np.asarray(features,float), np.asarray(weight,float)
    logits = features @ weight.T + bias
    if method == 'none':
        return logits
    a,b,p = fitted['source_mean'], fitted['target_mean'], fitted['basis']
    logits += (b-a) @ weight.T
    if method != 'mean_shift':
        matrix = fitted['matrices'][method]
        logits += ((features-a) @ p) @ (matrix-np.eye(fitted['rank'])) @ (p.T@weight.T)
    return logits


def apply_map(features, fitted, method):
    if method == 'none':
        return np.asarray(features,float).copy()
    a,b,p = fitted['source_mean'], fitted['target_mean'], fitted['basis']
    out = np.asarray(features,float)-a
    if method != 'mean_shift':
        out += (out@p) @ (fitted['matrices'][method]-np.eye(fitted['rank'])) @ p.T
    return out+b


def head_logits(fit_features, fit_truth, test_features, vocabulary, fitted, alpha=1.):
    """Supervised linear head: equal weight per labelled calibration worm.

    Full output vocabulary is retained, including classes absent in calibration.
    The encoder stays frozen. Unlike geometric correction, this uses taxon labels.
    """
    fit_truth = np.asarray(fit_truth,int)
    valid = fit_truth >= 0
    if not valid.any():
        raise ValueError('not_estimable: no labelled calibration individuals for head')
    p,a,scale = fitted['basis'],fitted['source_mean'],fitted['scale']
    x = (np.asarray(fit_features,float)[valid]-a) @ p / scale
    y = np.eye(vocabulary)[fit_truth[valid]]
    model = Ridge(alpha=alpha, fit_intercept=True).fit(x,y)
    return model.predict((np.asarray(test_features,float)-a) @ p / scale)


def aggregate_probabilities(logits, specimen_indices, n_specimens):
    prob = softmax(logits,axis=1)
    sums = np.zeros((n_specimens,prob.shape[1]))
    np.add.at(sums,specimen_indices,prob)
    counts = np.bincount(specimen_indices,minlength=n_specimens)
    means = sums / np.maximum(counts[:,None],1)
    pred = means.argmax(axis=1)
    pred[counts==0] = -1
    return pred,means


def matched_removal(train, species, stage, repeat, seed=2026):
    """Match removed worm count within species, keeping both training stages.

    Image-count differences are reported, not concealed. Validation/test are
    unchanged. Class weights are frozen separately by the training wrapper.
    """
    selected = train.loc[train.species_label.eq(species)]
    info = selected.groupby('barcode').agg(stage=('life_stage','first'),images=('filename','size'))
    removed = info.index[info.stage.eq(stage)].tolist()
    if not removed or info.stage.nunique()!=2 or len(info)-len(removed)<2:
        raise ValueError('not_estimable: a size-matched control cannot retain both stages')
    rng = np.random.default_rng(seed+repeat)
    target_images = int(info.loc[removed,'images'].sum())
    best = None
    for _ in range(2000):
        candidate = rng.choice(info.index, len(removed), replace=False).tolist()
        retained = info.drop(candidate)
        if retained.stage.nunique()!=2:
            continue
        mismatch = abs(int(info.loc[candidate,'images'].sum())-target_images)
        if best is None or mismatch<best[0]:
            best = mismatch, sorted(candidate)
    if best is None:
        raise ValueError('not_estimable: no control preserving both stages')
    return sorted(removed),best[1],best[0]
