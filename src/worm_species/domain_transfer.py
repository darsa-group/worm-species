"""Leakage-controlled paired-camera correction experiments on frozen features."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

METHODS = ("none", "gray_world", "mean_shift", "coral", "paired_ridge")


def stable_seed(*parts):
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8], 16)


def make_splits(metadata, folds=5, seed=2026):
    """Keep whole worms together; retain a separate split of their camera views."""
    frame = metadata.loc[metadata.domain.isin(["gphoto2", "webcam"])].copy()
    if frame.image_id.duplicated().any():
        raise ValueError("Duplicate images in split input")
    paired = set(frame.loc[frame.domain.eq("gphoto2"), "individual_id"]) & set(frame.loc[frame.domain.eq("webcam"), "individual_id"])
    frame = frame.loc[frame.individual_id.isin(paired)]
    individuals, views = [], {}
    for individual, group in frame.groupby("individual_id", sort=True):
        for column in ("biological_group", "group_kind", "life_stage"):
            if group[column].fillna("").nunique() != 1:
                raise ValueError(f"Conflicting {column}: {individual}")
        individuals.append({"individual_id": individual, "biological_group": group.iloc[0].biological_group,
                            "group_kind": group.iloc[0].group_kind, "life_stage": group.iloc[0].life_stage})
        views[individual] = {}
        for camera in ("gphoto2", "webcam"):
            ids = sorted(group.loc[group.domain.eq(camera), "image_id"])
            rng = np.random.default_rng(stable_seed(seed, individual, camera))
            rng.shuffle(ids)
            cut = max(1, len(ids)//2)
            views[individual][camera] = {"fit": sorted(ids[:cut]), "test": sorted(ids[cut:])}
    worms = pd.DataFrame(individuals)
    worms["fold"] = -1
    for (taxon, stage), group in worms.groupby(["biological_group", "life_stage"]):
        indices = group.index.to_numpy().copy()
        rng = np.random.default_rng(stable_seed(seed, taxon, stage))
        rng.shuffle(indices)
        offset = int(rng.integers(folds))
        for i, index in enumerate(indices):
            worms.loc[index, "fold"] = (i+offset) % folds
    if len(worms) < folds or set(worms.fold) != set(range(folds)):
        raise ValueError("Too few paired individuals for the requested folds")
    return worms, views


def make_trials(worms, views, cfg):
    trials = []
    all_ids = set(worms.individual_id)
    by_group = {name: set(group.individual_id) for name, group in worms.groupby("biological_group")}

    def add(protocol, fit, test, **extra):
        if not fit or not test:
            return
        if protocol != "within_individual" and set(fit) & set(test):
            raise ValueError("Correction calibration and evaluation individuals overlap")
        trials.append({"id": len(trials), "protocol": protocol, "fit": sorted(fit), "test": sorted(test),
                       "view_split": False, "fold": -1, "repeat": 0, "donor": "", "train_group": "all",
                       "test_group": "all", "calibration_size": "all", **extra})

    for fold in sorted(worms.fold.unique()):
        test = set(worms.loc[worms.fold.eq(fold), "individual_id"])
        fit = all_ids-test
        add("global", fit, test, fold=int(fold))
        for source_group, source_ids in by_group.items():
            for target_group, target_ids in by_group.items():
                add("within_species" if source_group == target_group else "across_species",
                    fit & source_ids, test & target_ids, fold=int(fold), train_group=source_group, test_group=target_group)
        for repeat in range(cfg["calibration_repeats"]):
            # Nested random samples, balanced round-robin across biological groups.
            rng = np.random.default_rng(stable_seed(cfg["seed"], int(fold), repeat))
            bins = [sorted(ids & fit) for ids in by_group.values()]
            for items in bins:
                rng.shuffle(items)
            ordered = []
            while any(bins):
                for i in rng.permutation(len(bins)):
                    if bins[i]:
                        ordered.append(bins[i].pop())
            for size in cfg["calibration_sizes"]:
                if size == "all":
                    if repeat:
                        continue
                    chosen = ordered
                elif int(size) <= len(ordered):
                    chosen = ordered[:int(size)]
                else:
                    continue
                add("calibration_curve", chosen, test, fold=int(fold), repeat=repeat, calibration_size=str(size))
    for name, ids in by_group.items():
        add("leave_species_out", all_ids-ids, ids, test_group=name)
    for row in worms.to_dict("records"):
        individual, taxon = row["individual_id"], row["biological_group"]
        if all(views[individual][c]["test"] for c in ("gphoto2", "webcam")):
            add("within_individual", [individual], [individual], donor=individual,
                train_group=taxon, test_group=taxon, view_split=True)
        add("individual_same_species", [individual], by_group[taxon]-{individual}, donor=individual,
            train_group=taxon, test_group=taxon, view_split=True)
        add("individual_other_species", [individual], all_ids-by_group[taxon], donor=individual,
            train_group=taxon, view_split=True)
    return trials


def image_indices(metadata, trial, camera, role, views):
    selected = metadata.domain.eq(camera) & metadata.individual_id.isin(trial[role])
    if trial["view_split"] and (role == "fit" or trial["protocol"] == "within_individual"):
        allowed = {image_id for worm in trial[role] for image_id in views[worm][camera][role]}
        selected &= metadata.image_id.isin(allowed)
    return np.flatnonzero(selected.to_numpy())


def centroids(features, metadata, indices, individuals):
    subset = metadata.iloc[indices]
    return np.stack([features[indices[subset.individual_id.to_numpy() == worm]].mean(axis=0) for worm in individuals])


@dataclass
class Correction:
    method: str
    source_mean: np.ndarray
    target_mean: np.ndarray
    basis: np.ndarray | None = None
    matrix: np.ndarray | None = None

    def apply(self, values):
        values = np.asarray(values, dtype=np.float64)
        if self.method in {"none", "gray_world"}:
            return values.copy()
        result = values-self.source_mean
        if self.basis is not None:
            projected = result @ self.basis
            result = result + (projected @ self.matrix-projected) @ self.basis.T
        return result+self.target_mean


def fit_correction(source, target, method, cfg):
    """Fit only paired calibration centroids. Class labels are never regression targets."""
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or not len(source):
        raise ValueError("Mismatched calibration centroids")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite calibration features")
    if method not in METHODS:
        raise ValueError("Unknown correction")
    a, b = source.mean(axis=0), target.mean(axis=0)
    correction = Correction(method, a, b)
    if method in {"none", "gray_world", "mean_shift"}:
        return correction
    if len(source) < 3:
        raise ValueError("not_estimable: covariance/ridge alignment needs at least three paired calibration individuals")
    # The bounded subspace makes repeated local fits feasible. The residual stays unchanged.
    rank = min(cfg["alignment_rank"], source.shape[1], 2*len(source)-2)
    centered = np.concatenate([source-a, target-b])
    if np.linalg.norm(centered) <= 1e-12:
        raise ValueError("not_estimable: calibration features have zero variation")
    basis = PCA(n_components=rank, svd_solver="randomized", random_state=cfg["seed"]).fit(centered).components_.T
    x, y = (source-a) @ basis, (target-b) @ basis
    scale = max(float(np.sqrt(np.mean(x*x))), 1e-8)
    x, y = x/scale, y/scale
    if method == "paired_ridge":
        # Penalize departure from identity, rather than shrinking all biology toward zero.
        matrix = np.eye(rank)+np.linalg.solve(x.T@x + cfg["ridge_alpha"]*len(x)*np.eye(rank), x.T@(y-x))
    else:
        def power(covariance, exponent):
            values, vectors = np.linalg.eigh(covariance)
            floor = max(float(np.trace(covariance)/rank)*cfg["covariance_floor"], 1e-10)
            return (vectors*np.maximum(values, floor)**exponent) @ vectors.T
        cx, cy = LedoitWolf().fit(x).covariance_, LedoitWolf().fit(y).covariance_
        matrix = power(cx, -.5) @ power(cy, .5)
    correction.basis, correction.matrix = basis, matrix
    return correction


def distances(source, target):
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    na, nb = np.linalg.norm(source, axis=1), np.linalg.norm(target, axis=1)
    valid = (na > 1e-12) & (nb > 1e-12)
    cosine = np.full(len(source), np.nan)
    cosine[valid] = np.clip(1-(source[valid]*target[valid]).sum(axis=1)/(na[valid]*nb[valid]), 0, 2)
    return cosine, np.sqrt(2*cosine)


def predictions(features, weights, bias):
    logits = np.asarray(features, dtype=float) @ np.asarray(weights, dtype=float).T + bias
    return logits, softmax(logits, axis=1)


def unit_predictions(logits, probabilities, truth, individual_ids, method):
    if method == "image":
        return np.asarray(truth), probabilities.argmax(axis=1), np.asarray(individual_ids)
    truths, guessed, worms = [], [], []
    for individual in sorted(set(individual_ids)):
        selected = np.asarray(individual_ids) == individual
        values = np.unique(np.asarray(truth)[selected])
        if len(values) != 1:
            raise ValueError("Conflicting labels within an individual")
        p, l = probabilities[selected], logits[selected]
        mean = p.mean(axis=0)
        if method == "mean_probabilities":
            scores = mean
        elif method == "mean_logits":
            scores = l.mean(axis=0)
        elif method == "median_probabilities":
            scores = np.median(p, axis=0)
        elif method == "majority_vote":
            counts = np.bincount(p.argmax(axis=1), minlength=p.shape[1])
            candidates = np.flatnonzero(counts == counts.max())
            winner = int(candidates[np.argmax(mean[candidates])])
            scores = np.zeros(p.shape[1]); scores[winner] = 1
        else:
            raise ValueError("Unknown aggregation")
        truths.append(int(values[0])); guessed.append(int(scores.argmax())); worms.append(individual)
    return np.array(truths), np.array(guessed), np.array(worms)


def score_counts(truth, predicted, classes):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    valid = truth >= 0
    cm = confusion_matrix(truth[valid], predicted[valid], labels=range(classes)) if valid.any() else np.zeros((classes, classes), dtype=int)
    return cm, int((~valid).sum())


def metrics_from_counts(cm):
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    truth_support, prediction_support = cm.sum(axis=1), cm.sum(axis=0)
    supported, union = truth_support > 0, (truth_support+prediction_support) > 0
    denominator = truth_support+prediction_support
    f1 = np.divide(2*cm.diagonal(), denominator, out=np.zeros(len(cm)), where=denominator>0)
    return {"n": int(n), "macro_f1": float(f1[union].mean()) if n else np.nan,
            "macro_f1_supported": float(f1[supported].mean()) if n else np.nan,
            "accuracy": float(cm.trace()/n) if n else np.nan,
            "balanced_accuracy": float((cm.diagonal()[supported]/truth_support[supported]).mean()) if n else np.nan,
            "classes_in_test": int(supported.sum()), "classes_in_model": len(cm), "uniform_chance": 1/len(cm)}
