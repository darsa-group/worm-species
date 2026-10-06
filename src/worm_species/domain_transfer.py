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

METHODS = ("none", "mean_shift", "coral", "paired_ridge", "procrustes", "head_ridge")

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

def distances(source, target):
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    na, nb = np.linalg.norm(source, axis=1), np.linalg.norm(target, axis=1)
    valid = (na > 1e-12) & (nb > 1e-12)
    cosine = np.full(len(source), np.nan)
    cosine[valid] = np.clip(1-(source[valid]*target[valid]).sum(axis=1)/(na[valid]*nb[valid]), 0, 2)
    return cosine, np.sqrt(2*cosine)
