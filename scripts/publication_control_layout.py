"""Matched-removal publication panel, using regenerated result tables."""

from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("work/publication")
TABLE = OUT / "supporting_data"


def figure():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    s = pd.read_csv(TABLE / "matched_stage_bootstrap_summary.csv")
    counts = pd.read_csv(TABLE / "matched_removed_counts.csv")
    keys = list(counts[["species", "stage"]].itertuples(index=False, name=None))
    fig = plt.figure(figsize=(11.5, 8.5), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 3])
    guide = fig.add_subplot(grid[0, :])
    guide.axis("off")
    guide.set_xlim(-0.02, 1.02)
    guide.text(
        0,
        1,
        "a  What changes in the training data?",
        fontweight="bold",
        fontsize=16,
        va="top",
    )
    for x, text in [
        (0.00, "Reference\nBoth stages present"),
        (0.34, "Remove one stage\nAll adults OR all juveniles\nof one species"),
        (0.68, "Random removal\nEqual worm/image counts\nBoth stages remain"),
    ]:
        guide.add_patch(
            FancyBboxPatch(
                (x, 0.32),
                0.30,
                0.45,
                boxstyle="round,pad=.012",
                facecolor="#ecf2f5",
                edgecolor="#7898a7",
            )
        )
        guide.text(x + 0.15, 0.55, text, ha="center", va="center", fontsize=12)
    guide.text(
        0,
        0.07,
        "Same outputs, weights and test worms; three fixed random-removal sets, each trained with 30 seeds.",
        fontsize=11,
    )
    labels = [
        f"{species.replace('Allolobophora_','All. ').replace('Aporrectodea_','Ap. ')} / {stage.lower()}\n{int(counts.iloc[i].removed_individuals)} worms, {int(counts.iloc[i].removed_images)} images removed"
        for i, (species, stage) in enumerate(keys)
    ]
    for col, task in enumerate(["genus", "species"]):
        ax = fig.add_subplot(grid[1, col])
        for offset, cohort, label, colour, marker in [
            (-0.20, "full_test", "All test stages", "#243c49", "o"),
            (0, "omitted_stage", "Stage removed from training", "#b7682d", "s"),
            (0.20, "retained_stage", "Stage retained in training", "#176b91", "o"),
        ]:
            q = (
                s[s.task.eq(task) & s.cohort.eq(cohort)]
                .set_index(["species", "stage_removed"])
                .loc[keys]
            )
            errors = np.vstack(
                [
                    q.excess_mean - q.excess_sampling_ci_low,
                    q.excess_sampling_ci_high - q.excess_mean,
                ]
            )
            ax.errorbar(
                q.excess_mean,
                np.arange(6) + offset,
                xerr=errors,
                fmt=marker,
                color=colour,
                capsize=3,
                markersize=7 if cohort == "full_test" else 5,
                markerfacecolor="white" if cohort == "retained_stage" else colour,
                elinewidth=1.8 if cohort == "full_test" else 1,
                label=label,
                zorder=4 if cohort == "full_test" else 3,
            )
        ax.axvline(0, color="#485965", linewidth=1.3, ls="--", zorder=1)
        for y in [0.5, 2.5, 4.5]:
            ax.axhspan(y, y + 1, facecolor="#f1f4f6", zorder=0)
        ax.tick_params(labelsize=12)
        ax.set(
            ylim=(5.7, -0.7),
            xlabel="Additional class F1 loss (percentage points)",
            yticks=range(6),
        )
        ax.set_yticklabels(labels if col == 0 else [""] * 6, fontsize=12)
        ax.set_title(
            ("b  " if col == 0 else "c  ")
            + task.capitalize()
            + " F1: paired removal contrast",
            fontweight="bold",
            fontsize=14,
            loc="left",
            pad=12,
        )
        ax.set_xlabel("Additional class F1 loss (percentage points)", fontsize=12)
        ax.grid(axis="x", alpha=0.12)
        if col == 1:
            ax.legend(
                fontsize=10, frameon=False, loc="lower left", bbox_to_anchor=(0, -0.23)
            )
    fig.suptitle(
        "ConvNeXt-Base: stage removal compared with equal-sized random removal",
        fontsize=14,
        fontweight="bold",
    )
    return fig
