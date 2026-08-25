#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml


@st.cache_data
def load_clusters(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_decisions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["image_id", "curation_label", "curation_notes"])
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def save_decision(path: Path, image_id: str, label: str, notes: str) -> None:
    decisions = load_decisions(path)
    decisions = decisions.loc[decisions["image_id"] != image_id].copy()
    decisions = pd.concat(
        [decisions, pd.DataFrame([{
            "image_id": image_id,
            "curation_label": label,
            "curation_notes": notes,
        }])],
        ignore_index=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    decisions.sort_values("image_id").to_csv(path, index=False)


def save_batch_decision(
    path: Path, image_ids: list[str], label: str, notes: str
) -> None:
    decisions = load_decisions(path)
    decisions = decisions.loc[~decisions["image_id"].isin(image_ids)].copy()
    additions = pd.DataFrame({
        "image_id": image_ids,
        "curation_label": label,
        "curation_notes": notes,
    })
    decisions = pd.concat([decisions, additions], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    decisions.sort_values("image_id").to_csv(path, index=False)


def main() -> None:
    st.set_page_config(page_title="GBIF Oligochaeta curation", layout="wide")
    with Path("configs/gbif_oligochaeta.yaml").open() as handle:
        config = yaml.safe_load(handle)
    workspace = config["workspace"]
    labels = config["curation"]["labels"]
    clusters_path = st.sidebar.text_input("Cluster manifest", workspace["clusters"])
    decisions_path = Path(st.sidebar.text_input("Decisions", workspace["decisions"]))
    if not Path(clusters_path).is_file():
        st.info("Run the embed and cluster commands before opening this app.")
        return
    frame = load_clusters(clusters_path)
    decisions = load_decisions(decisions_path)
    frame = frame.merge(decisions, on="image_id", how="left")
    frame["curation_label"] = frame["curation_label"].replace("", pd.NA).fillna("unreviewed")
    frame["curation_notes"] = frame["curation_notes"].fillna("")

    st.title("GBIF Oligochaeta DINOv3 cluster review")
    cluster_values = sorted(frame["cluster"].unique(), key=lambda value: int(value))
    chosen_clusters = st.sidebar.multiselect("Clusters", cluster_values, default=cluster_values)
    chosen_labels = st.sidebar.multiselect("Labels", labels, default=labels)
    genus_values = sorted(value for value in frame["genus"].unique() if value)
    chosen_genera = st.sidebar.multiselect("Genera", genus_values)
    filtered = frame.loc[
        frame["cluster"].isin(chosen_clusters)
        & frame["curation_label"].isin(chosen_labels)
    ].copy()
    if chosen_genera:
        filtered = filtered.loc[filtered["genus"].isin(chosen_genera)]
    st.caption(f"Showing {len(filtered):,} of {len(frame):,} images")

    st.sidebar.subheader("Batch decision")
    batch_cluster = st.sidebar.selectbox("Cluster to label", chosen_clusters)
    batch_label = st.sidebar.selectbox(
        "Cluster decision",
        [label for label in labels if label != "unreviewed"],
    )
    batch_notes = st.sidebar.text_input("Cluster decision notes")
    cluster_ids = frame.loc[frame["cluster"] == batch_cluster, "image_id"].tolist()
    st.sidebar.caption(f"This will update {len(cluster_ids):,} images in cluster {batch_cluster}.")
    if st.sidebar.button("Apply to entire cluster", disabled=not cluster_ids):
        save_batch_decision(decisions_path, cluster_ids, batch_label, batch_notes)
        st.cache_data.clear()
        st.rerun()

    plot = px.scatter(
        filtered,
        x="projection_x",
        y="projection_y",
        color="cluster",
        symbol="curation_label",
        hover_data=["image_id", "genus", "species", "scientific_name"],
        render_mode="webgl",
    )
    st.plotly_chart(plot, use_container_width=True)

    if filtered.empty:
        return
    page_size = st.sidebar.slider("Images per page", 8, 64, 24, 8)
    page_count = max(1, (len(filtered) + page_size - 1) // page_size)
    page = st.sidebar.number_input("Page", 1, page_count, 1)
    page_frame = filtered.iloc[(page - 1) * page_size: page * page_size]
    columns = st.columns(4)
    for position, (_, row) in enumerate(page_frame.iterrows()):
        with columns[position % 4]:
            if Path(row["local_path"]).is_file():
                st.image(row["local_path"], caption=f"{row['image_id']} | C{row['cluster']}")
            st.caption(f"{row['genus']} — {row['species'] or row['scientific_name']}")

    selected = st.selectbox(
        "Selected image",
        page_frame["image_id"].tolist(),
        format_func=lambda image_id: (
            f"{image_id} — "
            f"{page_frame.loc[page_frame['image_id'] == image_id, 'scientific_name'].iloc[0]}"
        ),
    )
    row = frame.loc[frame["image_id"] == selected].iloc[0]
    left, right = st.columns([1, 1])
    with left:
        if Path(row["local_path"]).is_file():
            st.image(row["local_path"])
    with right:
        current = row["curation_label"]
        label = st.radio("Decision", labels, index=labels.index(current) if current in labels else 0)
        notes = st.text_area("Notes", value=row["curation_notes"])
        if st.button("Save decision", type="primary"):
            save_decision(decisions_path, selected, label, notes)
            st.cache_data.clear()
            st.rerun()
        st.link_button("Open publisher image", row["source_url"])
        if row.get("media_reference"):
            st.link_button("Open media reference", row["media_reference"])
        st.write({
            "GBIF id": row["gbif_id"],
            "cluster": row["cluster"],
            "cluster probability": row["cluster_probability"],
            "license": row["license"],
            "creator": row["creator"],
        })

    decisions = load_decisions(decisions_path)
    export = frame.drop(columns=["curation_label", "curation_notes"]).merge(
        decisions, on="image_id", how="left"
    )
    export["curation_label"] = export["curation_label"].replace("", pd.NA).fillna("unreviewed")
    if st.sidebar.button("Export curated manifest"):
        destination = Path(workspace["curated_manifest"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        export.to_csv(destination, index=False)
        st.sidebar.success(f"Saved {destination}")


if __name__ == "__main__":
    main()
