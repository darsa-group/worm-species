#!/usr/bin/env python3
"""Generate a standalone, read-only notebook for the combined publication dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent


def build_notebook():
    cells = []

    def markdown(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"})

    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": dedent(text).strip() + "\n"})

    markdown("""
    # Publication dataset: composition, cameras and segmentation

    This notebook reads the combined **2025 + 2026 dataset**. It never modifies
    images, CSVs, segmentation records, or the running segmentation process.
    Use **Run All** to refresh the snapshot. During processing, the global CSV
    updates every 25 attempts; this is an artifact snapshot, not live process status.

    Analyses: dataset size · taxa and life stages · locations · camera coverage ·
    segmentation progress and failures · collection dates and weights · historical
    splits · sampled image properties · raw/mask previews · paired camera examples.

    **Counting units:** an image is one photograph; an individual is a unique
    `individual_id`. Two cameras photographing the same worm do not create two
    individuals. Calibration images are reported separately. `taxon` is the recorded
    label, which can be uncertain or a species complex; it is not automatically a
    verified species identification. No classifier accuracy is calculated here.
    """)
    markdown("""
    ## 1. Settings

    Edit the path and sample sizes here. All tabular composition analyses use the
    complete metadata snapshot. Image-property checks and galleries use small,
    deterministic samples to avoid competing with the running segmentation job.
    Dependencies: pandas, numpy, matplotlib, Pillow and IPython (already available
    in the wormspecies environment).
    """)
    code("""
    from pathlib import Path
    from datetime import datetime, timezone
    import hashlib
    import io
    import json
    import os

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    from IPython.display import display, Markdown

    DATASET_ROOT = Path(os.environ.get(
        "PUBLICATION_DATASET_ROOT", "/mnt/extssd/Earthworms/publication_dataset"
    )).expanduser().resolve()
    RANDOM_SEED = 20260922
    IMAGE_SAMPLE_PER_CAMERA = 24
    GALLERY_EXAMPLES = 3
    RUN_IMAGE_CHECKS = True
    SHOW_GALLERIES = True
    PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
    SUCCESS = {"reused", "segmented"}

    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 40)
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10})
    print("Dataset:", DATASET_ROOT)
    """)
    markdown("""
    ## 2. Load one coherent metadata snapshot

    Read codes as strings so `04` and `09` retain their leading zeros. All progress
    tables below come from this same snapshot, even if segmentation advances while
    the notebook is executing. If the main CSV has not yet been written, the
    preparation inventory is shown with status **not yet published**.
    """)
    code("""
    manifest_path = DATASET_ROOT / "metadata/images.csv"
    if manifest_path.is_file():
        with manifest_path.open("rb") as handle:
            payload = handle.read()
            manifest_mtime = os.fstat(handle.fileno()).st_mtime
        images = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
        snapshot_source = "metadata/images.csv"
    else:
        inventory_path = DATASET_ROOT / "metadata/inventory.json"
        if not inventory_path.is_file():
            raise FileNotFoundError(f"No dataset metadata in {DATASET_ROOT}. Check DATASET_ROOT or run make dataset-prepare.")
        with inventory_path.open("rb") as handle:
            payload = handle.read()
            manifest_mtime = os.fstat(handle.fileno()).st_mtime
        images = pd.DataFrame(json.loads(payload)["rows"]).fillna("").astype(str)
        images["segmentation_status"] = "not_yet_published"
        images["segmentation_error"] = ""
        snapshot_source = "metadata/inventory.json (preparation may still be running)"

    required = {"image_id", "individual_id", "barcode", "year", "camera", "kind", "taxon",
                "life_stage", "location_code", "capture_id", "raw_path", "segmented_path",
                "segmentation_status", "original_paper_split", "dataset_role"}
    missing = required - set(images.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    for column in ("timestamp", "weight_g", "raw_mask_path", "crop_mask_path", "segmentation_error"):
        if column not in images:
            images[column] = ""
    images["cohort"] = images["year"] + " / " + images["camera"]
    images["location_display"] = images["location_code"].replace("", "Not recorded")
    images["segmentation_ok"] = images["segmentation_status"].isin(SUCCESS)
    images["captured_at"] = pd.to_datetime(images["timestamp"], errors="coerce", utc=True, format="mixed")
    images["weight_numeric"] = pd.to_numeric(images["weight_g"], errors="coerce")
    worms = images.loc[images["kind"].eq("worm")].copy()
    non_worms = images.loc[~images["kind"].eq("worm")].copy()
    if worms.empty:
        raise ValueError("This snapshot contains no worm images to analyze.")
    # One row per worm, retaining conflicts explicitly instead of choosing an arbitrary label.
    def labels(values):
        return " | ".join(sorted(set(str(v) for v in values if str(v))))
    individuals = worms.groupby("individual_id", as_index=False).agg(
        year=("year", labels), barcode=("barcode", labels), taxon=("taxon", labels),
        life_stage=("life_stage", labels), locations=("location_display", labels),
        images=("image_id", "size"), cameras=("camera", "nunique")
    )
    snapshot = {
        "read_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": snapshot_source,
        "source_modified_utc": datetime.fromtimestamp(manifest_mtime, timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "all_images": len(images), "worm_images": len(worms),
        "unique_worms": worms["individual_id"].nunique(), "other_images": len(non_worms),
    }
    display(pd.Series(snapshot, name="Snapshot").to_frame())
    print("Status counts:", images["segmentation_status"].value_counts().to_dict())
    """)
    code("""
    def heatmap(table, title, xlabel="", ylabel="", decimals=0, cmap="Blues"):
        if table.empty:
            print(title + ": no data")
            return
        fig, ax = plt.subplots(figsize=(max(6, len(table.columns)*1.1), max(3, len(table)*0.38)))
        values = table.to_numpy(dtype=float)
        plot = ax.imshow(values, aspect="auto", cmap=cmap)
        ax.set(xticks=range(len(table.columns)), xticklabels=table.columns,
               yticks=range(len(table.index)), yticklabels=table.index,
               title=title, xlabel=xlabel, ylabel=ylabel)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        finite = values[np.isfinite(values)]
        midpoint = (finite.min() + finite.max()) / 2 if finite.size else 0
        if values.size <= 350:
            for (i, j), value in np.ndenumerate(values):
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.{decimals}f}", ha="center", va="center",
                            color="white" if value > midpoint else "black", fontsize=8)
        fig.colorbar(plot, ax=ax, fraction=.03, pad=.03)
        plt.tight_layout()
        plt.show()

    def sample_by_camera(frame, n):
        parts = [g.sample(min(n, len(g)), random_state=RANDOM_SEED)
                 for _, g in frame.sort_values("image_id").groupby("cohort", sort=True)]
        return pd.concat(parts, ignore_index=True) if parts else frame.iloc[:0].copy()

    def dataset_path(relative):
        if not str(relative).strip():
            raise FileNotFoundError("No path recorded")
        path = (DATASET_ROOT / str(relative)).resolve()
        if not path.is_relative_to(DATASET_ROOT):
            raise ValueError(f"Path leaves dataset folder: {relative}")
        return path

    def thumbnail(relative, max_size=(600, 450), is_mask=False, orient=True):
        with Image.open(dataset_path(relative)) as source:
            image = ImageOps.exif_transpose(source) if orient else source.copy()
            image = image.convert("L" if is_mask else "RGB")
            image.thumbnail(max_size, Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS)
            return image.copy()
    """)
    markdown("""
    ## 3. How many images and biological individuals?

    Individuals are counted separately within each camera column. **Do not add the
    camera totals** to obtain the overall number of worms; the same worm can occur
    in both. The year table deduplicates across cameras. Repeated captures increase
    image counts without increasing specimen counts.
    """)
    code("""
    composition = worms.groupby(["year", "camera"]).agg(
        images=("image_id", "size"), individuals=("individual_id", "nunique"),
        recorded_taxa=("taxon", "nunique"), segmented_images=("segmentation_ok", "sum")
    )
    composition["images_per_individual"] = (composition["images"] / composition["individuals"]).round(2)
    by_year = worms.groupby("year").agg(images=("image_id", "size"), individuals=("individual_id", "nunique"))
    display(composition)
    display(by_year)
    display(images.groupby(["year", "camera", "kind"]).size().rename("images").to_frame())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    composition["images"].plot.bar(ax=axes[0], color=PALETTE, title="Photographs per year and camera")
    composition["individuals"].plot.bar(ax=axes[1], color=PALETTE, title="Unique worms per year and camera")
    for ax in axes:
        ax.set_xlabel(""); ax.tick_params(axis="x", labelrotation=30)
    axes[0].set_ylabel("Images"); axes[1].set_ylabel("Individuals")
    plt.tight_layout(); plt.show()
    display(worms.groupby(["cohort", "individual_id"]).size().groupby("cohort").describe().round(2))
    """)
    markdown("""
    ## 4. Recorded taxa and developmental stages

    The first table counts images; the second counts unique individuals across
    cameras. Ambiguous names such as `Lumbricus_sp` and taxonomic complexes are
    retained as recorded. Differences here can change how difficult the external
    test set is; these are composition differences, not model performance.
    """)
    code("""
    taxon_images = pd.crosstab(worms["taxon"], worms["cohort"])
    taxon_individuals = pd.crosstab(individuals["taxon"], individuals["year"])
    display(taxon_images)
    display(taxon_individuals)
    heatmap(taxon_individuals, "Unique individuals per recorded taxon and year", "Year", "Recorded taxon")
    proportions = taxon_individuals.div(taxon_individuals.sum(axis=0), axis=1).mul(100)
    proportions.plot.barh(figsize=(10, max(4, len(proportions)*.4)), color=PALETTE,
                         title="Taxon composition by year (unique individuals)")
    plt.xlabel("Percent of individuals in that year"); plt.ylabel("")
    plt.tight_layout(); plt.show()
    stage_counts = pd.crosstab(individuals["taxon"], [individuals["year"], individuals["life_stage"]])
    stage_counts.columns = [f"{year} / {stage}" for year, stage in stage_counts.columns]
    heatmap(stage_counts, "Adults and juveniles by taxon (unique individuals)")
    years = sorted(worms["year"].unique())
    if len(years) > 1:
        historical_taxa = set(worms.loc[worms["year"].ne(years[-1]), "taxon"])
        latest_taxa = set(worms.loc[worms["year"].eq(years[-1]), "taxon"])
        print("Only in latest year:", sorted(latest_taxa - historical_taxa))
        print("Historical taxa absent from latest year:", sorted(historical_taxa - latest_taxa))
    uncertain_labels = sorted(t for t in worms["taxon"].unique() if len(t.split("_")) != 2 or "sp" in t.split("_"))
    print("Recorded labels needing taxonomic interpretation:", uncertain_labels)
    """)
    markdown("""
    ## 5. Locations and possible sampling imbalance

    Codes remain strings. Historical locations were not recorded in this combined
    manifest; they are shown as missing, never inferred. A specimen recorded at
    multiple locations contributes to each corresponding location count, so these
    totals may not be additive. No geographic coordinates are inferred from codes.
    """)
    code("""
    location_counts = worms.groupby(["year", "location_display"]).agg(
        images=("image_id", "size"), individuals=("individual_id", "nunique"))
    display(location_counts)
    known_locations = worms.loc[worms["location_code"].ne("")]
    if not known_locations.empty:
        location_taxa = known_locations.groupby(["taxon", "location_code"])["individual_id"].nunique().unstack(fill_value=0)
        heatmap(location_taxa, "Unique individuals by taxon and location", "Location code", "Recorded taxon")
        display(known_locations.groupby(["location_code", "camera"]).agg(
            images=("image_id", "size"), individuals=("individual_id", "nunique")))
    multiple_locations = worms.loc[worms["location_code"].ne("")].groupby("individual_id")["location_code"].nunique()
    print("Individuals recorded at multiple locations:", int(multiple_locations.gt(1).sum()))
    """)
    markdown("""
    ## 6. Camera coverage and repeated photographs

    Pair cameras by biological individual. Image numbers can differ between
    cameras; matching image numbers is not assumed to mean matching photographs.
    Session-level coverage below only establishes a shared capture session.
    """)
    code("""
    new_worms = worms.loc[worms["camera"].isin(["gphoto2", "webcam"])].copy()
    camera_counts = new_worms.groupby(["individual_id", "camera"]).size().unstack(fill_value=0)
    camera_counts = camera_counts.reindex(columns=["gphoto2", "webcam"], fill_value=0)
    if not camera_counts.empty:
        both = camera_counts.gt(0).all(axis=1)
        coverage = pd.Series({
            "Both cameras": int(both.sum()),
            "gphoto2 only": int((camera_counts.gphoto2.gt(0) & camera_counts.webcam.eq(0)).sum()),
            "webcam only": int((camera_counts.webcam.gt(0) & camera_counts.gphoto2.eq(0)).sum()),
        }, name="individuals")
        display(coverage.to_frame())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        coverage.plot.bar(ax=axes[0], color=PALETTE[:3], rot=0, title="Specimen coverage")
        axes[0].set_ylabel("Individuals")
        axes[1].scatter(camera_counts.gphoto2, camera_counts.webcam, alpha=.4, color=PALETTE[0])
        high = max(camera_counts.max().max(), 1)
        axes[1].plot([0, high], [0, high], "--", color="gray")
        axes[1].set(xlabel="gphoto2 images per worm", ylabel="webcam images per worm", title="Photo counts for the same specimen")
        plt.tight_layout(); plt.show()
        sessions = new_worms.groupby(["individual_id", "capture_id"])["camera"].nunique()
        print("Capture sessions containing both cameras:", int(sessions.eq(2).sum()), "/", len(sessions))
        display(camera_counts.loc[~both].head(20))
    else:
        print("No new camera rows yet.")
    """)
    markdown("""
    ## 7. Segmentation progress, failures and coverage

    **Reused** means an existing historical segmentation was copied. **Segmented**
    means the new workflow produced an output. **Pending** is not a failure.
    Attempt success is `segmented / (segmented + failed)` and excludes reused
    outputs and pending images. It is an operational success rate, **not an accuracy
    score for the mask**. Compare camera rates only after both have been processed.
    """)
    code("""
    status_order = ["reused", "segmented", "failed", "pending", "not_yet_published"]
    status = pd.crosstab(worms["cohort"], worms["segmentation_status"]).reindex(columns=status_order, fill_value=0)
    progress = status.copy()
    progress["total_images"] = status.sum(axis=1)
    progress["attempted_new"] = status.segmented + status.failed
    progress["attempt_success_pct"] = status.segmented.div(progress.attempted_new.replace(0, np.nan)).mul(100).round(1)
    progress["available_pct"] = (status.segmented + status.reused).div(progress.total_images).mul(100).round(1)
    display(progress)
    status.plot.barh(stacked=True, figsize=(11, 4), color=[PALETTE[2], PALETTE[0], PALETTE[5], "#CCCCCC", "#AAAAAA"])
    plt.xlabel("Worm images"); plt.ylabel(""); plt.title("Segmentation state at the snapshot time")
    plt.tight_layout(); plt.show()
    successful = worms.loc[worms["segmentation_ok"]]
    availability = worms.groupby("cohort")["individual_id"].nunique().rename("all_individuals").to_frame()
    availability["individuals_with_output"] = successful.groupby("cohort")["individual_id"].nunique().reindex(availability.index, fill_value=0)
    availability["individuals_without_output"] = availability.all_individuals - availability.individuals_with_output
    display(availability)
    attempts = worms.loc[worms["segmentation_status"].isin(["segmented", "failed"])].copy()
    if not attempts.empty:
        attempts["failed"] = attempts["segmentation_status"].eq("failed")
        failure_rates = attempts.groupby(["cohort", "taxon", "life_stage"]).agg(
            attempted=("image_id", "size"), failed=("failed", "sum"))
        failure_rates["failure_pct"] = (100 * failure_rates.failed / failure_rates.attempted).round(1)
        display(failure_rates.sort_values(["failed", "attempted"], ascending=False).head(25))
    failed = worms.loc[worms["segmentation_status"].eq("failed")]
    display(failed[["image_id", "cohort", "barcode", "location_code", "raw_path", "segmentation_error"]].head(25))
    """)
    markdown("""
    ## 8. Collection dates and specimen weights

    Dates are parsed in UTC. Daily specimen counts are unique within each day;
    the same worm can appear on several days. Weight plots use **one median weight
    per biological individual**, combining repeated camera records. Nonpositive
    or missing weights are excluded and counted explicitly.
    """)
    code("""
    dated = worms.loc[worms["captured_at"].notna()].copy()
    dated["date"] = dated["captured_at"].dt.date
    daily = dated.groupby(["date", "cohort"]).agg(images=("image_id", "size"), individuals=("individual_id", "nunique"))
    display(daily.tail(20))
    if not daily.empty:
        daily["individuals"].unstack(fill_value=0).plot.bar(stacked=True, figsize=(12, 4), color=PALETTE)
        plt.ylabel("Individuals per day and camera"); plt.xlabel("UTC capture date")
        plt.title("Capture schedule (camera counts overlap)"); plt.tight_layout(); plt.show()
    print("Images with unparseable timestamps:", int(worms.captured_at.isna().sum()))
    print("Image rows with missing/nonpositive weights:", int((worms.weight_numeric.isna() | worms.weight_numeric.le(0)).sum()))
    weight_rows = worms.loc[worms.weight_numeric.gt(0)]
    weights = weight_rows.groupby("individual_id")["weight_numeric"].median().rename("weight_g").to_frame()
    weights = weights.join(individuals.set_index("individual_id")[["year", "taxon", "life_stage"]])
    if not weights.empty:
        display(weights.groupby(["year", "life_stage"]).weight_g.describe().round(3))
        fig, ax = plt.subplots(figsize=(10, 4))
        for (year, stage), group in weights.groupby(["year", "life_stage"]):
            ax.hist(group.weight_g, bins=np.linspace(0, weights.weight_g.max()*1.01, 25),
                    histtype="step", linewidth=1.8, label=f"{year} / {stage} (n={len(group)})")
        ax.set(xlabel="Median recorded weight per individual (g)", ylabel="Individuals", title="Specimen weight distribution")
        ax.legend(); plt.tight_layout(); plt.show()
    """)
    markdown("""
    ## 9. Identity, label and original split checks

    These checks inspect metadata, not a new train/test split. Historical images
    without paper membership remain unassigned. A repeated barcode across years
    requires identity review; a new year prefix alone cannot prove independence.
    Camera copies of one specimen must stay together in any future training split.
    """)
    code("""
    label_conflicts = worms.groupby("individual_id")[["taxon", "life_stage"]].nunique()
    label_conflicts = label_conflicts.loc[label_conflicts.gt(1).any(axis=1)]
    duplicate_ids = images.loc[images.image_id.duplicated(keep=False)]
    duplicate_paths = images.loc[images.raw_path.duplicated(keep=False)]
    cross_year_barcodes = worms.groupby("barcode").year.nunique()
    cross_year_barcodes = cross_year_barcodes.loc[cross_year_barcodes.gt(1)]
    split_rows = worms.loc[worms.original_paper_split.ne("")]
    split_membership = split_rows.groupby("individual_id").original_paper_split.nunique()
    split_conflicts = split_membership.loc[split_membership.gt(1)]
    checks = pd.Series({
        "duplicate_image_id_rows": len(duplicate_ids),
        "duplicate_raw_path_rows": len(duplicate_paths),
        "individuals_with_conflicting_taxon_or_stage": len(label_conflicts),
        "barcodes_present_in_multiple_years": len(cross_year_barcodes),
        "individuals_in_multiple_original_paper_splits": len(split_conflicts),
        "worms_with_missing_taxon": int(worms.taxon.eq("").sum()),
        "worms_with_missing_life_stage": int(worms.life_stage.eq("").sum()),
    }, name="count")
    display(checks.to_frame())
    display(worms.groupby("dataset_role").agg(images=("image_id", "size"), individuals=("individual_id", "nunique")))
    display(split_rows.groupby("original_paper_split").agg(images=("image_id", "size"), individuals=("individual_id", "nunique")))
    if not label_conflicts.empty: display(label_conflicts)
    if not split_conflicts.empty: display(split_conflicts)
    if not cross_year_barcodes.empty: display(cross_year_barcodes.to_frame("years"))
    if not duplicate_ids.empty: display(duplicate_ids[["image_id", "raw_path"]])
    """)
    markdown("""
    ## 10. Sampled image dimensions, brightness and mask coverage

    This section samples up to `IMAGE_SAMPLE_PER_CAMERA` images per year/camera.
    It reports dimensions, file size and mean grayscale intensity. These are
    descriptive checks; lighting, framing and specimen content differ, so they
    do not establish which camera is scientifically better. Mask coverage is
    measured on a reduced mask preview. Historical masks are compared with the
    raw image, **never with the differently sized segmented crop**.
    """)
    code("""
    image_properties = pd.DataFrame()
    if RUN_IMAGE_CHECKS:
        property_rows = []
        for row in sample_by_camera(worms, IMAGE_SAMPLE_PER_CAMERA).to_dict("records"):
            record = {"image_id": row["image_id"], "cohort": row["cohort"], "barcode": row["barcode"], "error": ""}
            try:
                path = dataset_path(row["raw_path"])
                with Image.open(path) as image:
                    width, height = image.size
                    if image.getexif().get(274) in {5, 6, 7, 8}: width, height = height, width
                preview = thumbnail(row["raw_path"], (256, 256))
                record.update(width=width, height=height, megapixels=width*height/1e6,
                              file_mb=path.stat().st_size/1e6,
                              mean_gray=float(np.asarray(preview.convert("L"), dtype=float).mean()))
                if row["segmentation_ok"]:
                    with Image.open(dataset_path(row["segmented_path"])) as crop:
                        record["segmented_megapixels"] = crop.width*crop.height/1e6
                    mask_preview = thumbnail(row["raw_mask_path"], (256, 256), is_mask=True, orient=False)
                    record["raw_mask_foreground_pct"] = 100*float((np.asarray(mask_preview)>0).mean())
                    if row["segmentation_status"] == "segmented":
                        with Image.open(dataset_path(row["crop_mask_path"])) as mask, \
                             Image.open(dataset_path(row["segmented_path"])) as crop:
                            record["crop_mask_aligned"] = mask.size == crop.size
            except (OSError, ValueError) as error:
                record["error"] = f"{type(error).__name__}: {error}"
            property_rows.append(record)
        image_properties = pd.DataFrame(property_rows)
        metrics = [c for c in ["megapixels", "file_mb", "mean_gray", "raw_mask_foreground_pct"] if c in image_properties]
        if metrics:
            display(image_properties.groupby("cohort")[metrics].agg(["count", "median", "min", "max"]).round(2))
            fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 4), squeeze=False)
            for ax, metric in zip(axes.flat, metrics):
                groups = [(name, group[metric].dropna()) for name, group in image_properties.groupby("cohort")]
                groups = [(name, values) for name, values in groups if len(values)]
                if groups:
                    ax.boxplot([v for _, v in groups], tick_labels=[n for n, _ in groups])
                ax.set_title(metric.replace("_", " ")); ax.tick_params(axis="x", labelrotation=40)
            plt.tight_layout(); plt.show()
        display(image_properties.loc[image_properties.error.ne("")])
    else:
        print("Image checks disabled; metadata analyses still use all rows.")
    """)
    markdown("""
    ## 11. Inspect original images, masks and segmented crops

    Each preview uses a mask in **raw-image coordinates**. The black-background
    crop is displayed separately. If an old mask's dimensions/orientation do not
    match the raw image, the overlay is skipped rather than resized into alignment.
    These examples help inspect masks; successful file creation alone does not
    demonstrate correct biological segmentation.
    """)
    code("""
    def raw_overlay(row):
        with Image.open(dataset_path(row["raw_path"])) as source:
            # Historical segmentation used the stored pixel orientation.
            raw = source.copy() if row["segmentation_status"] == "reused" else ImageOps.exif_transpose(source)
            raw = raw.convert("RGB")
        with Image.open(dataset_path(row["raw_mask_path"])) as source:
            mask = source.convert("L")
        if raw.size != mask.size:
            raise ValueError(f"Raw/mask dimensions differ: {raw.size} vs {mask.size}")
        raw.thumbnail((600, 450))
        mask = mask.resize(raw.size, Image.Resampling.NEAREST)
        pixels = np.asarray(raw).copy()
        foreground = np.asarray(mask)>0
        pixels[foreground] = (.65*pixels[foreground] + .35*np.array([0, 200, 255])).astype(np.uint8)
        return pixels

    def show_segmentation_examples(frame, n=3):
        selected = sample_by_camera(frame, n)
        if selected.empty:
            print("No successful segmentation examples in this snapshot."); return
        for cohort, group in selected.groupby("cohort"):
            fig, axes = plt.subplots(len(group), 3, figsize=(12, 3.2*len(group)), squeeze=False)
            fig.suptitle(cohort)
            for axes_row, row in zip(axes, group.to_dict("records")):
                previews = [("Raw", lambda: thumbnail(row["raw_path"])),
                            ("Mask overlay", lambda: raw_overlay(row)),
                            ("Segmented", lambda: thumbnail(row["segmented_path"], orient=False))]
                for ax, (title, get_image) in zip(axes_row, previews):
                    try: ax.imshow(get_image())
                    except (OSError, ValueError) as error: ax.text(.03, .5, str(error), wrap=True, fontsize=8)
                    ax.set_title(f"{title}: {row['barcode']}", fontsize=8); ax.axis("off")
            plt.tight_layout(); plt.show()
    if SHOW_GALLERIES:
        show_segmentation_examples(worms.loc[worms.segmentation_ok], GALLERY_EXAMPLES)
    """)
    markdown("""
    ## 12. Compare the cameras on the same specimen; inspect failures

    These are representative photos of the same specimen in a shared capture
    session, **not necessarily simultaneous exposures**. They help separate
    camera appearance from differences in which worms were sampled. Failed-image
    previews are for manual review; this notebook does not modify their status.
    """)
    code("""
    if SHOW_GALLERIES and not new_worms.empty:
        shared_sessions = new_worms.groupby(["individual_id", "capture_id"])["camera"].nunique()
        keys = sorted(shared_sessions.loc[shared_sessions.eq(2)].index)
        rng = np.random.default_rng(RANDOM_SEED)
        chosen = [keys[i] for i in rng.choice(len(keys), min(GALLERY_EXAMPLES, len(keys)), replace=False)] if keys else []
        for individual_id, capture_id in chosen:
            subset = new_worms.loc[new_worms.individual_id.eq(individual_id) & new_worms.capture_id.eq(capture_id)]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for ax, camera in zip(axes, ["gphoto2", "webcam"]):
                row = subset.loc[subset.camera.eq(camera)].sort_values("raw_path").iloc[0]
                try: ax.imshow(thumbnail(row.raw_path))
                except (OSError, ValueError) as error: ax.text(.03, .5, str(error), wrap=True)
                ax.set_title(f"{camera} — {row.segmentation_status}"); ax.axis("off")
            fig.suptitle(f"{individual_id} / {capture_id}", fontsize=10)
            plt.tight_layout(); plt.show()
        failed_examples = sample_by_camera(failed, GALLERY_EXAMPLES)
        if not failed_examples.empty:
            fig, axes = plt.subplots(len(failed_examples), 1, figsize=(9, 3*len(failed_examples)), squeeze=False)
            for ax, row in zip(axes.flat, failed_examples.to_dict("records")):
                try: ax.imshow(thumbnail(row["raw_path"]))
                except (OSError, ValueError) as error: ax.text(.03, .5, str(error), wrap=True)
                ax.set_title(f"Failed: {row['camera']} / {row['barcode']} / location {row['location_display']}", fontsize=9)
                ax.axis("off")
            plt.tight_layout(); plt.show()
    """)
    markdown("""
    ## Refreshing and interpreting the notebook

    Use **Run All** to reload the latest published metadata and refresh every view.
    Values above describe the snapshot timestamp shown in section 2. A pending
    count does not establish that the process is still running, and a low failure
    count during an incomplete run does not establish the final failure rate.

    Useful follow-up questions: Are taxa or life stages concentrated at particular
    locations? Are some worms missing a camera? Are segmentation failures concentrated
    in one taxon or camera? Do any specimens have no usable segmented image? Are
    new taxa absent from the historical dataset? These checks should inform later
    classifier evaluation while keeping the new gphoto2 cohort as external test.
    """)
    for index, cell in enumerate(cells):
        cell["id"] = f"dataset-{index:02d}"
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }, "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "notebooks/publication_dataset_overview.ipynb")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_notebook(), indent=1) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
