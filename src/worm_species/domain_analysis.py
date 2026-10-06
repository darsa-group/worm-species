"""Individual prediction aggregation and paired-camera representation statistics."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score, confusion_matrix

METHODS = ("mean_probabilities", "mean_logits", "majority_vote", "median_probabilities")


def aggregate_predictions(frame, methods=METHODS):
    """Aggregate one model/seed/domain/input/task without mixing observational units."""
    for column in ("model", "seed", "domain", "input", "task"):
        if frame[column].nunique() != 1:
            raise ValueError(f"Aggregation requires exactly one {column}")
    if frame.image_id.duplicated().any() or frame.individual_id.eq("").any():
        raise ValueError("Duplicate images or missing individual IDs")
    if not set(methods) <= set(METHODS):
        raise ValueError("Unknown aggregation method")
    output = []
    for individual, group in frame.groupby("individual_id", sort=True):
        if group.true_index.nunique() != 1 or group.true_label.fillna("").nunique() != 1:
            raise ValueError(f"Conflicting task labels for individual {individual}")
        probabilities = np.array([json.loads(value) for value in group.probabilities_json], dtype=float)
        logits = np.array([json.loads(value) for value in group.logits_json], dtype=float)
        if (probabilities.ndim != 2 or logits.shape != probabilities.shape
                or not np.isfinite(probabilities).all() or not np.isfinite(logits).all()
                or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
            raise ValueError("Invalid probability/logit vectors")
        mean = probabilities.mean(axis=0)
        base = group.iloc[0].to_dict()
        base.update(image_id="", individual_id=individual, n_images=len(group), prediction_unit="individual")
        for method in methods:
            if method == "mean_probabilities":
                scores = mean
            elif method == "mean_logits":
                scores = softmax(logits.mean(axis=0))
            elif method == "median_probabilities":
                scores = np.median(probabilities, axis=0)
                scores = scores / scores.sum() if scores.sum() else mean
            else:
                votes = np.bincount(probabilities.argmax(axis=1), minlength=len(mean))
                scores = votes / votes.sum()
            candidates = np.flatnonzero(scores == scores.max())
            # Majority ties: highest mean probability, then lowest checkpoint class index.
            winner = int(candidates[np.argmax(mean[candidates])]) if method == "majority_vote" else int(scores.argmax())
            output.append({**base, "method": method, "predicted_index": winner,
                           "probabilities_json": json.dumps(scores.tolist()), "logits_json": ""})
    return pd.DataFrame(output)


def metrics(frame, labels):
    """Keep reference-compatible macro-F1 and also a true-support-fixed macro-F1."""
    scored = frame.loc[frame.true_index >= 0]
    truth = scored.true_index.to_numpy(dtype=int)
    predicted = scored.predicted_index.to_numpy(dtype=int)
    k = len(labels)
    if k < 1 or ((truth >= k) | (predicted < 0) | (predicted >= k)).any():
        raise ValueError("Prediction outside checkpoint vocabulary")
    cm = confusion_matrix(truth, predicted, labels=range(k)) if len(truth) else np.zeros((k, k), dtype=int)
    supported = np.flatnonzero(cm.sum(axis=1))
    return {"n": len(scored), "unscored": len(frame)-len(scored), "classes_in_model": k,
            "classes_in_test": len(supported), "uniform_chance": 1/k,
            "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)) if len(truth) else None,
            "macro_f1_supported": float(f1_score(truth, predicted, labels=supported, average="macro", zero_division=0)) if len(truth) else None,
            "accuracy": float((truth == predicted).mean()) if len(truth) else None,
            "balanced_accuracy": float((cm.diagonal()[supported]/cm.sum(axis=1)[supported]).mean()) if len(truth) else None}, cm


def unit_vectors(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Embeddings must be a finite 2D matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError("Zero-norm embedding cannot define a cosine distance")
    return values / norms


def paired_representation(metadata, features):
    """Mean raw features per individual/camera, then L2-normalize the centroids.

    Return one statistical row per paired worm. Different-individual comparisons
    average over all other paired worms within camera, plus same-taxon controls.
    """
    if len(metadata) != len(features) or metadata.image_id.duplicated().any():
        raise ValueError("Embedding rows do not uniquely align with metadata")
    frame = metadata.reset_index(drop=True)
    present = [set(frame.loc[frame.domain.eq(camera), "individual_id"]) for camera in ("gphoto2", "webcam")]
    paired = sorted(present[0] & present[1])
    if len(paired) < 2:
        raise ValueError("At least two paired individuals are required")
    centroids, within, counts, tags = {}, {}, {}, {}
    for camera in ("gphoto2", "webcam"):
        raw_means, scatter, image_counts = [], [], []
        for individual in paired:
            group = frame.loc[frame.domain.eq(camera) & frame.individual_id.eq(individual)]
            for field in ("taxon", "genus", "life_stage"):
                if group[field].nunique(dropna=False) != 1:
                    raise ValueError(f"Inconsistent {field} for {individual}")
            tag = tuple(group.iloc[0][field] for field in ("taxon", "genus", "life_stage"))
            if individual in tags and tags[individual] != tag:
                raise ValueError(f"Conflicting camera metadata for {individual}")
            tags[individual] = tag
            raw = np.asarray(features)[group.index]
            mean = raw.mean(axis=0)
            center = unit_vectors(mean[None, :])[0]
            normalized = unit_vectors(raw)
            # One image does not estimate within-individual variation.
            scatter.append(float(np.clip(1-normalized @ center, 0, 2).mean()) if len(raw)>1 else np.nan)
            raw_means.append(mean)
            image_counts.append(len(raw))
        centroids[camera] = unit_vectors(raw_means)
        within[camera], counts[camera] = scatter, image_counts
    g, w = centroids["gphoto2"], centroids["webcam"]
    cosine = np.clip(1-np.sum(g*w, axis=1), 0, 2)
    rows = []
    matrices = {camera: np.clip(1-z @ z.T, 0, 2) for camera, z in centroids.items()}
    for i, individual in enumerate(paired):
        taxon, genus, stage = tags[individual]
        row = {"individual_id": individual, "taxon": taxon, "genus": genus, "life_stage": stage,
               "paired_cosine_distance": cosine[i], "paired_normalized_euclidean": np.linalg.norm(g[i]-w[i])}
        for camera in ("gphoto2", "webcam"):
            other = np.arange(len(paired)) != i
            same_taxon = np.array([tags[p][0] == taxon and tags[p][2] == stage for p in paired]) & other
            row.update({f"{camera}_images": counts[camera][i],
                        f"{camera}_within_individual_cosine": within[camera][i],
                        f"{camera}_different_individual_cosine": float(matrices[camera][i, other].mean()),
                        f"{camera}_same_taxon_stage_other_cosine": float(matrices[camera][i, same_taxon].mean()) if same_taxon.any() else np.nan,
                        f"{camera}_same_taxon_stage_other_n": int(same_taxon.sum())})
        row["different_individual_same_camera_cosine"] = np.mean([row[f"{c}_different_individual_cosine"] for c in ("gphoto2", "webcam")])
        row["camera_minus_different_individual_cosine"] = cosine[i]-row["different_individual_same_camera_cosine"]
        rows.append(row)
    points = pd.DataFrame([{"individual_id": p, "domain": camera, "taxon": tags[p][0],
                            "genus": tags[p][1], "life_stage": tags[p][2]}
                           for camera in ("gphoto2", "webcam") for p in paired])
    return pd.DataFrame(rows), points, np.concatenate([g, w])


def bootstrap_mean(values, seed=2026, samples=2000):
    """Descriptive interval resampling paired individuals, never image pairs."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"individuals": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)])
    low, high = np.quantile(means, [.025, .975])
    return {"individuals": len(values), "mean": float(values.mean()), "ci95_low": float(low), "ci95_high": float(high)}
