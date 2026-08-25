from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..models.factory import resolve_dinov3_model_name
from .pipeline import resolve_manifest_image_path


class ManifestImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row["_resolved_local_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, str(row["image_id"])


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def embed_manifest(
    manifest_path: str | Path,
    embeddings_path: str | Path,
    index_path: str | Path,
    *,
    model_name: str,
    batch_size: int,
    num_workers: int,
    device_name: str = "auto",
    l2_normalize: bool = True,
) -> dict:
    try:
        import timm
        from timm.data import create_transform, resolve_data_config
    except ImportError as exc:
        raise RuntimeError(
            "DINOv3 embedding requires timm>=1.0.20; install the GBIF environment"
        ) from exc

    canonical = resolve_dinov3_model_name(model_name)
    if canonical is None:
        raise ValueError(f"Unsupported DINOv3 model: {model_name}")
    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if "download_status" in frame:
        frame = frame.loc[frame["download_status"] == "downloaded"].copy()
    frame["_resolved_local_path"] = frame["local_path"].map(
        lambda value: str(resolve_manifest_image_path(manifest_path, value))
    )
    frame = frame.loc[
        frame["_resolved_local_path"].map(lambda value: Path(value).is_file())
    ].copy()
    if frame.empty:
        raise ValueError("No downloaded images are available for embedding")

    model = timm.create_model(canonical, pretrained=True, num_classes=0)
    model.eval()
    device = _device(device_name)
    model.to(device)
    data_config = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**data_config, is_training=False)
    loader = DataLoader(
        ManifestImageDataset(frame, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    chunks: list[np.ndarray] = []
    image_ids: list[str] = []
    with torch.inference_mode():
        for images, batch_ids in loader:
            features = model(images.to(device, non_blocking=True))
            if isinstance(features, (tuple, list)):
                features = features[0]
            if features.ndim > 2:
                features = features.flatten(2).mean(dim=-1)
            if l2_normalize:
                features = torch.nn.functional.normalize(features.float(), dim=1)
            chunks.append(features.float().cpu().numpy())
            image_ids.extend(batch_ids)
    embeddings = np.concatenate(chunks, axis=0)
    output = Path(embeddings_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, embeddings)
    os.replace(temporary, output)
    index = frame.drop(columns=["_resolved_local_path"]).set_index("image_id")
    index = index.loc[image_ids].reset_index()
    index.insert(1, "embedding_row", np.arange(len(index), dtype=int))
    index_output = Path(index_path)
    index_output.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(index_output, index=False)
    metadata = {
        "model_alias": model_name,
        "canonical_model": canonical,
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
        "device": str(device),
        "rows": int(embeddings.shape[0]),
        "dimensions": int(embeddings.shape[1]),
        "l2_normalized": bool(l2_normalize),
        "pretrained_data_config": data_config,
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n"
    )
    return metadata


def cluster_embeddings(
    embeddings_path: str | Path,
    index_path: str | Path,
    output_path: str | Path,
    *,
    seed: int,
    pca_dimensions: int,
    projection: str,
    min_cluster_size: int,
    min_samples: int,
) -> dict:
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA

    embeddings = np.load(embeddings_path)
    index = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    if len(index) != len(embeddings):
        raise ValueError("Embedding array and index have different row counts")
    dimensions = max(1, min(pca_dimensions, len(embeddings), embeddings.shape[1]))
    reduced = PCA(n_components=dimensions, random_state=seed).fit_transform(embeddings)
    if projection == "umap":
        try:
            # Some conda builds need an explicit writable cache location before
            # importing numba-backed UMAP modules.
            os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/worm-gbif-numba-cache")
            Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
            import umap
        except Exception as exc:
            raise RuntimeError("projection=umap requires a working umap-learn install") from exc
        coordinates = umap.UMAP(
            n_components=2,
            random_state=seed,
            transform_seed=seed,
            n_jobs=1,
        ).fit_transform(reduced)
    elif projection == "pca":
        if reduced.shape[1] == 1:
            coordinates = np.column_stack([reduced[:, 0], np.zeros(len(reduced))])
        else:
            coordinates = reduced[:, :2]
    else:
        raise ValueError("projection must be 'pca' or 'umap'")

    if len(embeddings) < min_cluster_size:
        labels = np.full(len(embeddings), -1, dtype=int)
        probabilities = np.zeros(len(embeddings), dtype=float)
    else:
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            copy=True,
        )
        labels = clusterer.fit_predict(reduced)
        probabilities = clusterer.probabilities_
    result = index.copy()
    result["projection_x"] = coordinates[:, 0]
    result["projection_y"] = coordinates[:, 1]
    result["cluster"] = labels
    result["cluster_probability"] = probabilities
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    summary = {
        "rows": int(len(result)),
        "clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
        "noise_rows": int(np.sum(labels == -1)),
        "projection": projection,
        "pca_dimensions": dimensions,
        "seed": seed,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
