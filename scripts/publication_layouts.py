"""Publication layouts adapted from the active manuscript's plotting source."""

from pathlib import Path
import json, sys, os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/publication"
DATA = ROOT / "work/results/analysis/tables"
ANALYSIS = ROOT / "work/results/analysis"
TASKS = ["age", "genus", "species"]
TASK_NAMES = ["Developmental stage", "Genus", "Species"]
COLOURS = ["#176b91", "#27837a", "#b7682d"]
DOMAINS = ["original", "gphoto2", "webcam"]
DOMAIN_NAMES = ["2025 Canon", "2026 Canon", "2026 USB"]
MODELS = ["convnext_base", "vit_b_16", "resnet50"]
METHODS = ["none", "mean_shift", "coral", "paired_ridge", "procrustes", "head_ridge"]
METHOD_NAMES = [
    "Uncorrected",
    "Mean shift",
    "CORAL",
    "Paired ridge",
    "Procrustes",
    "Supervised head ridge",
]
METHOD_COLOURS = ["#73848d", "#d99b20", "#176b91", "#27837a", "#9263ab", "#bd5372"]


def read(name):
    return pd.read_csv(
        DATA / name if (DATA / name).is_file() else OUT / "supporting_data" / name
    )


def replace_stream_labels(fig):
    for text in fig.findobj(matplotlib.text.Text):
        value = text.get_text()
        for old, new in [
            ("2025HQ", "2025 Canon"),
            ("2026HQ", "2026 Canon"),
            ("2026LQ", "2026 USB"),
        ]:
            value = value.replace(old, new)
        value = value.replace(
            "HQ: higher-resolution Canon stream   |   LQ: lower-resolution USB-camera stream",
            "Canon: reference acquisition   |   USB: alternate acquisition",
        )
        value = value.replace("2026 Canon/LQ", "2026 Canon/USB")
        value = value.replace(
            "2025 Canon and 2026 Canon\nSame Canon system\n2026 USB: USB camera",
            "Canon: 2025 and 2026\nSame imaging system\nUSB: 2026 alternate",
        )
        value = value.replace("2025 Canon development", "2025 reference images")
        value = value.replace(
            "2025 Canon / 2026 Canon / 2026 USB", "Reference / Canon / USB"
        )
        value = value.replace(
            "Original classifier held fixed", "Original encoder held fixed"
        )
        value = value.replace(
            "Remove a species–stage group\nRetrain with that group excluded\nRetain the independent test worms",
            "Exclude a stage from training\nCompare matched random removal\nKeep validation/test unchanged",
        )
        value = value.replace(
            "Genus / species F1; absent output noted",
            "Genus / species F1; both test stages",
        )
        value = value.replace("Fit feature calibration", "Fit calibration")
        value = value.replace(
            "2026 USB → 2026 Canon\nCORAL or paired ridge\nCalibrate on paired worm means",
            "USB → Canon calibration\nDistributional / paired / head\nFit on separate worm means",
        )
        text.set_text(value)


def savefig(fig, name):
    replace_stream_labels(fig)
    fig.savefig(OUT / "figures" / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figures" / f"{name}.png", bbox_inches="tight", dpi=180)
    if name == "figure1_workflow":
        with plt.rc_context({"svg.fonttype": "none"}):
            fig.savefig(OUT / "figures" / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def visual_figure(legacy):
    """Paper-width panels: type sizes are verified after PDF embedding."""
    visual = read("visual_task_summary.csv")
    chance = read("chance_controls.csv")
    tensor = legacy.specimen_tensor(
        pd.read_csv(OUT / "supporting_data/figure2_example_source.csv").iloc[0]
    )
    fig = plt.figure(figsize=(6.4, 7.3), layout="constrained")
    grid = fig.add_gridspec(3, 2)
    recipes = [
        (
            "a",
            "Shape, greyscale and colour",
            "representations",
            [
                ("silhouette", 0, "Silhouette", "#73848d"),
                ("grey", 0, "Greyscale", "#73848d"),
                ("original", 0, "Colour", "#73848d"),
            ],
        ),
        (
            "b",
            "Fine spatial detail",
            "gaussian_blur_percent",
            [
                ("blur", 0, "Unmodified", "#73848d"),
                ("blur", 50, "50% blur", "#73848d"),
                ("blur", 100, "100% blur", "#73848d"),
            ],
        ),
        (
            "c",
            "Spatial resolution",
            "resolution_loss",
            [
                ("resolution", 0, "224 px", "#73848d"),
                ("resolution", 97, "7 px", "#73848d"),
                ("resolution", 99, "2 px", "#73848d"),
            ],
        ),
        (
            "d",
            "Spatial organisation",
            "patch_shuffle",
            [
                ("patch", 2, "2 × 2", "#73848d"),
                ("patch", 8, "8 × 8", "#73848d"),
                ("patch", 16, "16 × 16", "#73848d"),
            ],
        ),
        (
            "e",
            "Combined changes: species",
            "combined",
            [
                ("blur_grey", 50, "Greyscale", "#73848d"),
                ("blur", 50, "Colour", "#73848d"),
                ("blur_patch", 50, "Shuffled", "#73848d"),
            ],
        ),
    ]
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    ):
        for index, (letter, title, kind, examples) in enumerate(recipes):
            card = grid[index // 2, index % 2].subgridspec(
                2, 1, height_ratios=[3.5, 1.6], hspace=0.08
            )
            ax = fig.add_subplot(card[0])
            ax.set_title(
                letter + "  " + title,
                loc="left",
                fontsize=10.5,
                fontweight="bold",
                pad=6,
            )
            for task in (TASKS if kind != "combined" else ["species"]):
                ti = TASKS.index(task)
                colour = COLOURS[ti]
                ref = visual[
                    (visual.task == task)
                    & (visual.condition == "resolution_loss_000pct")
                ]["mean"].iloc[0]
                c = chance[
                    (chance.domain == "original")
                    & (chance.task == task)
                    & (
                        chance.score_set
                        == ("fixed_eight" if task == "species" else "full")
                    )
                ].chance_mean.iloc[0]
                if kind == "combined":
                    for recipe, label, style, colour in [
                        ("colour", "Blur only", "-", COLOURS[0]),
                        ("grey", "Blur → grey", "--", COLOURS[1]),
                        ("patch", "Blur → shuffle", "-.", COLOURS[2]),
                    ]:
                        conditions = {
                            "colour": [
                                "resolution_loss_000pct",
                                "gaussian_blur_050pct",
                                "gaussian_blur_075pct",
                            ],
                            "grey": [
                                "colour_000pct",
                                "blur050__colour000",
                                "blur075__colour000",
                            ],
                            "patch": [
                                "patch_shuffle_16x16",
                                "blur050__patch16x16",
                                "blur075__patch16x16",
                            ],
                        }[recipe]
                        q = (
                            visual[
                                (visual.task == task)
                                & visual.condition.isin(conditions)
                            ]
                            .set_index("condition")
                            .loc[conditions]
                        )
                        ax.errorbar(
                            [0, 50, 75],
                            q["mean"],
                            q.seed_sd,
                            fmt="o",
                            linestyle=style,
                            color=colour,
                            capsize=2,
                            ms=3.5,
                            label=label,
                        )
                    ax.set_xticks([0, 50, 75])
                    ax.set_xlabel("Blur severity (%)")
                    ax.legend(
                        fontsize=9.5,
                        frameon=False,
                        loc="lower left",
                        handlelength=1.3,
                        labelspacing=0.2,
                        borderpad=0.15,
                    )
                elif kind == "representations":
                    conditions = [
                        "binary_mask_only",
                        "colour_000pct",
                        "resolution_loss_000pct",
                    ]
                    q = (
                        visual[
                            (visual.task == task) & visual.condition.isin(conditions)
                        ]
                        .set_index("condition")
                        .loc[conditions]
                    )
                    ax.errorbar(
                        np.arange(3) + (ti - 1) * 0.08,
                        q["mean"],
                        q.seed_sd,
                        fmt="o-",
                        color=colour,
                        capsize=2,
                        ms=3.5,
                    )
                    ax.set_xticks(range(3), ["Silhouette", "Greyscale", "Colour"])
                else:
                    q = visual[
                        (visual.task == task) & (visual["transform"] == kind)
                    ].copy()
                    if kind == "resolution_loss":
                        q.loc[q.condition == "resolution_loss_000pct", "level"] = 224
                    q = q.sort_values("level")
                    ax.errorbar(
                        q.level,
                        q["mean"],
                        q.seed_sd,
                        fmt="o-",
                        color=colour,
                        capsize=2,
                        ms=3.5,
                    )
                    if kind == "resolution_loss":
                        ax.set_xscale("log")
                        ax.set_xticks([1, 7, 28, 224], ["1", "7", "28", "224"])
                    if kind == "patch_shuffle":
                        ax.set_xticks([2, 4, 8, 16])
                    if kind == "gaussian_blur_percent":
                        ax.set_xticks([0, 25, 50, 75, 100])
                    ax.set_xlabel(
                        {
                            "resolution_loss": "Retained side (pixels)",
                            "patch_shuffle": "Patch-grid side",
                            "gaussian_blur_percent": "Blur severity (%)",
                        }[kind]
                    )
                ax.axhline(ref, color=COLOURS[ti], ls="--", lw=1, alpha=0.7)
                ax.axhline(c, color=COLOURS[ti], ls=":", lw=1, alpha=0.8)
            ax.set_ylim(0, 1.02)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.tick_params(labelsize=10)
            if index % 2 == 0:
                ax.set_ylabel("Individual macro-F1", fontsize=10)
            else:
                ax.tick_params(labelleft=False)
            ax.grid(axis="y", alpha=0.15)
            montage = fig.add_subplot(card[1])
            montage.axis("off")
            for ei, (operation, level, label, _) in enumerate(examples):
                label = label.replace("Blur + ", "Blur +\n")
                montage.text(
                    (ei + 0.5) / 3,
                    0.99,
                    label,
                    ha="center",
                    va="top",
                    fontsize=9.5,
                    color="#536b78",
                    transform=montage.transAxes,
                )
                image_ax = montage.inset_axes([ei / 3 + 0.025, -0.05, 0.283, 0.76])
                image_ax.imshow(
                    legacy.visual_transform(tensor, operation, level)
                    .permute(1, 2, 0)
                    .numpy()
                )
                image_ax.axis("off")
        legend = fig.add_subplot(grid[2, 1])
        legend.axis("off")
        handles = [
            Line2D([0], [0], marker="o", color=c, label=t)
            for c, t in zip(COLOURS, TASK_NAMES)
        ]
        handles += [
            Line2D([0], [0], color="black", ls="--", label="Unmodified task F1"),
            Line2D([0], [0], color="black", ls=":", label="Simulated chance F1"),
        ]
        legend.legend(
            handles=handles,
            loc="upper left",
            frameon=False,
            fontsize=10,
            labelspacing=0.7,
            borderaxespad=0.5,
        )
        legend.text(
            0.05,
            0.09,
            "Mean of 30 seeds; bars: 1 SD\nOne example worm throughout",
            ha="left",
            va="bottom",
            fontsize=9.5,
            transform=legend.transAxes,
        )
    return fig


def external_figure(legacy):
    summary = read("baseline_summary.csv")
    chance = read("chance_controls.csv")
    fig = plt.figure(figsize=(6.4, 7.1), layout="constrained")
    gs = fig.add_gridspec(3, 6, height_ratios=[1.2, 1.9, 1.65])
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10,
        }
    ):
        guide = fig.add_subplot(gs[0, :])
        guide.axis("off")
        guide.set(xlim=(0, 1), ylim=(0, 1))
        guide.set_title(
            "a  Reference development and external evaluation",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
        )
        for i, label in enumerate(["Train / select", "Freeze model", "Evaluate"]):
            guide.text(
                (i + 0.5) / 3,
                0.96,
                label,
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=.25", fc="#f3f7f9", ec="#a6bac3"),
            )
        for x in [0.29, 0.62]:
            guide.annotate(
                "",
                xy=(x + 0.08, 0.91),
                xytext=(x, 0.91),
                arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#263c47"),
            )
        rows = pd.read_csv(OUT / "supporting_data/figure4_example_sources.csv")
        for i, (_, row) in enumerate(rows.iterrows()):
            ax = guide.inset_axes([0.015 + i / 3, 0.17, 0.30, 0.45])
            ax.imshow(legacy.raw_photograph(row))
            ax.axis("off")
            guide.text(
                (i + 0.5) / 3,
                0.69,
                DOMAIN_NAMES[i],
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
            guide.text(
                (i + 0.5) / 3,
                0.09,
                [
                    "1,274 images / 221 worms",
                    "1,622 images / 296 worms",
                    "1,460 images / 293 worms",
                ][i],
                ha="center",
                fontsize=9.5,
            )
        guide.text(1 / 6, -0.03, "Different 2025 worm", ha="center", fontsize=9.5)
        guide.text(
            2 / 3, -0.03, "Same 2026 worm; different views", ha="center", fontsize=9.5
        )
        for ti, (task, title) in enumerate(zip(TASKS, ["Stage", "Genus", "Species"])):
            ax = fig.add_subplot(gs[1, 2 * ti : 2 * ti + 2])
            ax.set_title(
                "bcd"[ti] + "  " + title, loc="left", fontweight="bold", fontsize=10.5
            )
            for j, model in enumerate(MODELS):
                q = (
                    summary[
                        (summary.model == model)
                        & (summary.task == task)
                        & (summary.cohort == "all_segmented")
                        & (
                            summary.score_set
                            == ("common_six" if task == "species" else "full")
                        )
                    ]
                    .set_index("domain")
                    .loc[DOMAINS]
                )
                err = np.maximum(
                    0,
                    np.array(
                        [q["mean"] - q.sampling_ci_low, q.sampling_ci_high - q["mean"]]
                    ),
                )
                ax.errorbar(
                    np.arange(3) + (j - 1) * 0.16,
                    q["mean"],
                    yerr=err,
                    fmt=["o", "s", "^"][j],
                    color=COLOURS[j],
                    capsize=2,
                    ms=4,
                    label=["ConvNeXt", "ViT", "ResNet"][j],
                )
            c = (
                chance[
                    (chance.task == task)
                    & (
                        chance.score_set
                        == ("common_six" if task == "species" else "full")
                    )
                ]
                .set_index("domain")
                .loc[DOMAINS]
            )
            ax.plot(range(3), c.chance_mean, "k:", label="Chance")
            q = (
                summary[
                    (summary.model == MODELS[0])
                    & (summary.task == task)
                    & (summary.cohort == "all_segmented")
                    & (
                        summary.score_set
                        == ("common_six" if task == "species" else "full")
                    )
                ]
                .set_index("domain")
                .loc[DOMAINS]
            )
            labels = [
                f'{name.replace(" ",chr(10))}\n{int(row.images):,}\n{int(row.individuals)}'
                for name, (_, row) in zip(DOMAIN_NAMES, q.iterrows())
            ]
            ax.set(
                xticks=range(3), xticklabels=labels, ylim=(0, 1.04), xlim=(-0.4, 2.4)
            )
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.tick_params(axis="x", labelsize=9.5)
            ax.set_xlabel("Images / worms", fontsize=9.5)
            if ti == 0:
                ax.set_ylabel("Individual macro-F1")
                ax.legend(
                    frameon=False,
                    fontsize=9.5,
                    loc="lower left",
                    handlelength=1,
                    labelspacing=0.3,
                )
            else:
                ax.tick_params(labelleft=False)
            ax.grid(axis="y", alpha=0.15)
        delta = read("domain_differences.csv")
        pair = fig.add_subplot(gs[2, :3])
        for j, model in enumerate(MODELS):
            q = (
                delta[
                    (delta.model == model) & delta.comparison.eq("webcam minus gphoto2")
                ]
                .set_index("task")
                .loc[TASKS]
            )
            err = np.array(
                [
                    q.difference_pp - q.sampling_ci_low_pp,
                    q.sampling_ci_high_pp - q.difference_pp,
                ]
            )
            pair.errorbar(
                q.difference_pp,
                np.arange(3) + (j - 1) * 0.16,
                xerr=err,
                fmt=["o", "s", "^"][j],
                capsize=2,
                ms=4,
                color=COLOURS[j],
            )
        pair.axvline(0, color="#485965", linewidth=1)
        pair.set(
            yticks=range(3),
            yticklabels=["Stage", "Genus", "Species"],
            xlabel="F1 change (percentage points)",
        )
        pair.set_title(
            "e  Paired USB minus Canon", loc="left", fontsize=10.5, fontweight="bold"
        )
        pair.grid(axis="x", alpha=0.15)
        confusion = fig.add_subplot(gs[2, 3:])
        pred = pd.read_csv(ANALYSIS / "baseline_predictions.csv.gz")
        d = pred[
            pred.model.eq("convnext_base")
            & pred.domain.eq("webcam")
            & pred.task.eq("genus")
        ]
        cm = np.zeros((3, 3))
        np.add.at(cm, (d.true_index, d.predicted_index), 1)
        cm = 100 * cm / cm.sum(1, keepdims=True)
        confusion.imshow(cm, vmin=0, vmax=100, cmap="Blues")
        names = ["All.", "Ap.", "L."]
        n = d.groupby("true_index").individual_id.nunique()
        confusion.set(
            xticks=range(3),
            xticklabels=names,
            yticks=range(3),
            yticklabels=[f"{name}\nn={n[i]}" for i, name in enumerate(names)],
            xlabel="Predicted genus",
            ylabel="True genus",
        )
        confusion.tick_params(labelsize=9.5)
        for y in range(3):
            for x in range(3):
                confusion.text(
                    x,
                    y,
                    f"{cm[y,x]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if cm[y, x] > 55 else "#263c47",
                )
        confusion.set_title(
            "f  USB genus: ConvNeXt", loc="left", fontsize=10.5, fontweight="bold"
        )
    return fig


def calibration_figure():
    g = read("global_calibration_sampling_summary.csv")
    curve = read("calibration_summary.csv")
    b = read("baseline_summary.csv")
    fig = plt.figure(figsize=(6.4, 6.7), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[0.85, 2, 2.9])
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    ):
        guide = fig.add_subplot(gs[0])
        guide.axis("off")
        guide.set(xlim=(0, 1), ylim=(0, 1))
        guide.text(
            0.01,
            0.99,
            "a  ConvNeXt-Base: five-fold calibration",
            fontweight="bold",
            fontsize=10.5,
            va="top",
        )
        for j, n in enumerate([58, 58, 61, 58, 58]):
            x = 0.01 + j * 0.14
            guide.add_patch(
                Rectangle(
                    (x, 0.28),
                    0.13,
                    0.41,
                    facecolor="#f5dfcb" if j == 0 else "#d7ece9",
                    edgecolor="white",
                )
            )
            guide.text(
                x + 0.065,
                0.485,
                f"Fold {j+1}\n{n} worms",
                ha="center",
                va="center",
                fontsize=10,
            )
        guide.text(0.76, 0.60, "Fit: 235 worms", fontsize=10, fontweight="bold")
        guide.text(0.76, 0.32, "Test: 58 worms", fontsize=10, fontweight="bold")
        guide.text(
            0.01,
            0.03,
            "Rotate test fold; keep both cameras and all views with each worm.",
            fontsize=10,
        )
        ax = fig.add_subplot(gs[1])
        ax.set_title(
            "b  Classification gain over uncorrected USB",
            loc="left",
            fontweight="bold",
            fontsize=10.5,
        )
        for i, task in enumerate(TASKS):
            q = g[g.task == task].set_index("method").loc[METHODS[1:]]
            err = np.maximum(
                0,
                np.array(
                    [
                        q.gain_pp - q.gain_sampling_ci_low_pp,
                        q.gain_sampling_ci_high_pp - q.gain_pp,
                    ]
                ),
            )
            ax.errorbar(
                np.arange(5) + (i - 1) * 0.16,
                q.gain_pp,
                yerr=err,
                fmt="o",
                capsize=2,
                ms=4,
                color=COLOURS[i],
                label=TASK_NAMES[i],
            )
            raw = g[(g.task == task) & (g.method == "none")]["mean"].iloc[0]
            hq = b[
                (b.model == MODELS[0])
                & (b.task == task)
                & (b.domain == "gphoto2")
                & (b.cohort == "paired_segmented")
                & (b.score_set == ("common_six" if task == "species" else "full"))
            ]["mean"].iloc[0]
            ax.axhline(100 * (hq - raw), color=COLOURS[i], ls=":", lw=1, alpha=0.8)
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.set_xticks(
            range(5),
            [
                "Mean shift",
                "CORAL",
                "Paired ridge",
                "Procrustes",
                "Supervised\nhead ridge",
            ],
        )
        ax.set_ylabel("Macro-F1 gain (pp)")
        ax.set_ylim(-10, 90)
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.grid(axis="y", alpha=0.15)
        ax.legend(
            frameon=False,
            fontsize=9.5,
            loc="upper left",
            ncol=3,
            handlelength=1,
            columnspacing=1,
        )
        ax.text(
            0.02,
            0.03,
            "Dotted: paired Canon gains; dashed: zero gain",
            transform=ax.transAxes,
            fontsize=9.5,
        )
        ax = fig.add_subplot(gs[2])
        ax.set_title(
            "c  Species F1 at fixed PCA rank eight",
            loc="left",
            fontweight="bold",
            fontsize=10.5,
        )
        for method, label, c in zip(
            METHODS[1:],
            [
                "Mean shift",
                "CORAL",
                "Paired ridge",
                "Procrustes",
                "Supervised head ridge",
            ],
            METHOD_COLOURS[1:],
        ):
            q = curve[
                (curve.protocol == "calibration_curve")
                & (curve.task == "species")
                & (curve.method == method)
            ].copy()
            q["n"] = q.calibration_size.replace({"all": "234"}).astype(int)
            q = q.sort_values("n")
            ax.plot(q.n, q["mean"], "o-", color=c, label=label, ms=4)
            ax.fill_between(
                q.n, q["mean"] - q.seed_sd, q["mean"] + q.seed_sd, color=c, alpha=0.09
            )
        raw = g[(g.task == "species") & (g.method == "none")]["mean"].iloc[0]
        hq = b[
            (b.model == MODELS[0])
            & (b.task == "species")
            & (b.domain == "gphoto2")
            & (b.cohort == "paired_segmented")
            & (b.score_set == "common_six")
        ]["mean"].iloc[0]
        chance = (
            read("chance_controls.csv")
            .query(
                "domain == 'webcam' and task == 'species' and score_set == 'common_six'"
            )
            .chance_mean.iloc[0]
        )
        for value, style, label in [
            (raw, "--", "Uncorrected USB"),
            (hq, ":", "Paired Canon"),
            (chance, "-.", "Simulated chance"),
        ]:
            ax.axhline(value, color="#485965", ls=style, lw=1, label=label)
        ax.set(
            xscale="log",
            ylim=(0, 0.85),
            xlabel="Calibration worms per fold",
            ylabel="Individual species macro-F1",
        )
        ax.set_xticks(
            [5, 10, 20, 50, 100, 234], ["5", "10", "20", "50", "100", "232–235"]
        )
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.grid(axis="y", alpha=0.15)
        ax.legend(
            frameon=False,
            fontsize=9.5,
            loc="upper left",
            ncol=3,
            handlelength=1.4,
            columnspacing=0.8,
            labelspacing=0.35,
        )
    return fig
