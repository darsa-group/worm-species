"""Workflow and example transformations from the active manuscript source."""

from pathlib import Path
import sys, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/publication"
DATA = ROOT / "data/earthworm-images"
BLUE = "#176b91"
ORANGE = "#b7682d"
TEAL = "#27837a"
GREY = "#73848d"
RED = "#b94c3d"


def specimen_tensor(row):
    import torch
    from torchvision.transforms import functional as tf

    torch.set_num_threads(1)
    source = getattr(row, "source_segmented", "")
    path = (
        Path(source)
        if isinstance(source, str) and source
        else DATA / row.segmented_path
    )
    if not path.is_file():
        path = DATA / row.segmented_path
    with Image.open(path) as image:
        return tf.resize(tf.to_tensor(image.convert("RGB")), [224, 224], antialias=True)


def raw_photograph(row):
    from PIL import ImageOps

    path = DATA / row.rel_path_raw
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1400, 1400))
        return np.asarray(image)


def make_figure1():
    """Three-stage workflow using cached MIT-licensed Tabler SVG icons as vectors."""
    import json
    import xml.etree.ElementTree as ET
    from fontTools.pens.basePen import BasePen
    from fontTools.svgLib.path import SVGPath
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import (
        FancyBboxPatch,
        FancyArrowPatch,
        Circle,
        PathPatch,
        Rectangle,
    )
    from matplotlib.transforms import Affine2D

    assets = OUT / "assets/tabler"
    manifest = json.loads((assets / "sources.json").read_text())
    assert manifest["collection"] == "Tabler Icons"
    fig, ax = plt.subplots(figsize=(13, 11.1))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.99)
    ax.set(xlim=(0, 15), ylim=(0, 12.7))
    ax.axis("off")
    ink = "#203b49"
    muted = "#566f7b"
    pale = "#f5f8fa"
    colours = [TEAL, BLUE, ORANGE]

    # SVG curves stay vector paths in PDF/SVG; original files and licence are cached.
    class PlotPen(BasePen):
        def __init__(self):
            super().__init__(None)
            self.vertices = []
            self.codes = []
            self.start = None

        def _moveTo(self, p):
            self.start = p
            self.vertices.append(p)
            self.codes.append(MplPath.MOVETO)

        def _lineTo(self, p):
            self.vertices.append(p)
            self.codes.append(MplPath.LINETO)

        def _curveToOne(self, p1, p2, p3):
            self.vertices.extend([p1, p2, p3])
            self.codes.extend([MplPath.CURVE4] * 3)

        def _qCurveToOne(self, p1, p2):
            self.vertices.extend([p1, p2])
            self.codes.extend([MplPath.CURVE3] * 2)

        def _closePath(self):
            self.vertices.append(self.start)
            self.codes.append(MplPath.CLOSEPOLY)

        def _endPath(self):
            pass

    def svg(name, x, y, size, color=ink):
        root = ET.parse(assets / (name + ".svg")).getroot()
        for element in list(root):
            if element.attrib.get("stroke") == "none":
                root.remove(element)
        pen = PlotPen()
        SVGPath.fromstring(ET.tostring(root)).draw(pen)
        patch = PathPatch(
            MplPath(pen.vertices, pen.codes),
            fill=False,
            edgecolor=color,
            lw=1.8,
            capstyle="round",
            joinstyle="round",
            transform=Affine2D().scale(size / 24, -size / 24).translate(x, y + size)
            + ax.transData,
        )
        ax.add_patch(patch)

    def arrow(start, end, color=GREY, **kwargs):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                color=color,
                lw=1.6,
                **kwargs,
            )
        )

    def box(x, y, w, h, colour, fill="white"):
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.04, y - 0.05),
                w,
                h,
                boxstyle="round,pad=.03,rounding_size=.12",
                facecolor="#e5edf0",
                edgecolor="none",
                zorder=0,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=.03,rounding_size=.12",
                facecolor=fill,
                edgecolor=colour,
                lw=1.1,
                zorder=1,
            )
        )

    def header(n, y, title, question, colour):
        ax.add_patch(Circle((0.40, y + 0.06), 0.20, facecolor=colour, edgecolor="none"))
        ax.text(
            0.40,
            y + 0.06,
            str(n),
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            0.74,
            y + 0.08,
            title,
            ha="left",
            va="center",
            color=colour,
            fontsize=15,
            fontweight="bold",
        )
        ax.text(
            0.74, y - 0.26, question, ha="left", va="center", color=muted, fontsize=11
        )

    def card(x, y, w, h, n, title, icon, colour, body):
        box(x, y, w, h, colour)
        ax.text(
            x + 0.17,
            y + h - 0.25,
            f"{n:02d}",
            color=colour,
            fontsize=10,
            fontweight="bold",
            va="center",
        )
        ax.text(
            x + 0.56,
            y + h - 0.25,
            title,
            color=ink,
            fontsize=12,
            fontweight="bold",
            va="center",
        )
        svg(icon, x + 0.17, y + h - 1.14, 0.66, colour)
        ax.text(
            x + 0.97,
            y + h - 0.63,
            body,
            ha="left",
            va="top",
            color=ink,
            fontsize=11,
            linespacing=1.55,
        )

    ax.text(
        0.19,
        12.33,
        "From specimen collection to calibrated classification",
        color=ink,
        fontsize=19,
        fontweight="bold",
        va="center",
    )
    ax.text(
        0.19,
        11.94,
        "A controlled test of visual information, biological coverage and acquisition-domain change",
        color=muted,
        fontsize=11.5,
        va="center",
    )
    # 1. A single specimen-linked dataset, with different collection and imaging roles.
    header(
        1,
        11.35,
        "Create the specimen-linked image dataset",
        "Collaborators collect and identify; we photograph specimens and construct the dataset.",
        TEAL,
    )
    y = 8.60
    w = 3.30
    h = 2.07
    xs = [0.20, 3.95, 7.70, 11.45]
    card(
        xs[0],
        y,
        w,
        h,
        1,
        "Collect & identify",
        "shovel",
        TEAL,
        "Soil excavation\nSieving / hand sorting\nTaxon + adult/juvenile",
    )
    svg("zoom-scan", xs[0] + 0.21, y + 0.23, 0.51, TEAL)
    card(
        xs[1],
        y,
        w,
        h,
        2,
        "Acquire images",
        "camera",
        TEAL,
        "2025HQ and 2026HQ\nSame Canon system\n2026LQ: USB camera",
    )
    svg("device-computer-camera", xs[1] + 0.21, y + 0.23, 0.51, TEAL)
    card(
        xs[2],
        y,
        w,
        h,
        3,
        "Segment & link",
        "photo-scan",
        TEAL,
        "Mask and crop\nBarcode-linked labels\nRecord failed masks",
    )
    card(
        xs[3],
        y,
        w,
        h,
        4,
        "Assemble dataset",
        "database",
        TEAL,
        "2025HQ development\n2026 paired cameras\nAll views retain worm ID",
    )
    for x, next_x in zip(xs[:-1], xs[1:]):
        arrow((x + w + 0.05, y + h / 2), (next_x - 0.07, y + h / 2), TEAL)
    ax.text(
        0.32,
        8.31,
        "HQ: higher-resolution Canon stream   |   LQ: lower-resolution USB-camera stream",
        color=muted,
        fontsize=10,
    )

    # 2. Two parallel controlled experiments branch from the reference classifier.
    header(
        2,
        7.75,
        "Establish classification, then test its requirements",
        "What image information and species–stage coverage support taxonomic recognition?",
        BLUE,
    )
    y = 4.65
    w = 4.38
    h = 2.36
    xs2 = [0.20, 5.20, 10.20]
    card(
        xs2[0],
        y,
        w,
        h,
        5,
        "Reference classification",
        "brain",
        BLUE,
        "Train / validation / test worms\n718 / 166 / 221 individuals\nThree architectures; 30 seeds",
    )
    ax.text(
        xs2[0] + 0.18,
        y + 0.28,
        "Stage, genus and species heads",
        fontsize=10,
        color=BLUE,
        fontweight="bold",
    )
    card(
        xs2[1],
        y,
        w,
        h,
        6,
        "Visual information",
        "photo",
        BLUE,
        "Silhouette → greyscale → colour\nBlur; resolution; patch shuffling\nCombined visual conditions",
    )
    ax.text(
        xs2[1] + 0.18,
        y + 0.28,
        "Retrain and evaluate each condition",
        fontsize=10,
        color=BLUE,
        fontweight="bold",
    )
    card(
        xs2[2],
        y,
        w,
        h,
        7,
        "Biological coverage",
        "table-minus",
        BLUE,
        "Remove a species–stage group\nRetrain with that group excluded\nRetain the independent test worms",
    )
    ax.text(
        xs2[2] + 0.18,
        y + 0.28,
        "Genus / species F1; absent output noted",
        fontsize=10,
        color=BLUE,
        fontweight="bold",
    )
    # The branch deliberately routes above both cards: visual and biological tests are parallel.
    branch_y = 7.20
    ax.plot([2.39, 2.39, 12.39], [7.01, branch_y, branch_y], color=BLUE, lw=1.4)
    arrow((7.39, branch_y), (7.39, 7.04), BLUE)
    arrow((12.39, branch_y), (12.39, 7.04), BLUE)
    ax.text(
        0.32,
        4.31,
        "The 2025HQ test cohort is held fixed; all photographs of a worm stay in one partition.",
        color=muted,
        fontsize=10,
    )

    # 3. Score another acquisition stream, fit on paired calibration worms, test other worms.
    header(
        3,
        3.76,
        "Evaluate a new acquisition system and calibrate it",
        "Does calibration improve classification on individuals excluded from correction fitting?",
        ORANGE,
    )
    y = 0.69
    h = 2.37
    card(
        xs2[0],
        y,
        w,
        h,
        8,
        "External evaluation",
        "chart-dots-3",
        ORANGE,
        "2025HQ / 2026HQ / 2026LQ\nFrozen encoder and heads\nFeatures + descriptive UMAP",
    )
    ax.text(
        xs2[0] + 0.18,
        y + 0.28,
        "2026HQ/LQ: identical individuals",
        fontsize=10,
        color=ORANGE,
        fontweight="bold",
    )
    card(
        xs2[1],
        y,
        w,
        h,
        9,
        "Fit feature calibration",
        "adjustments-horizontal",
        ORANGE,
        "2026LQ → 2026HQ\nCORAL or paired ridge\nCalibrate on paired worm means",
    )
    ax.text(
        xs2[1] + 0.18,
        y + 0.28,
        "Five folds; calibration worms ≠ test worms",
        fontsize=10,
        color=ORANGE,
        fontweight="bold",
    )
    card(
        xs2[2],
        y,
        w,
        h,
        10,
        "Test held-out decisions",
        "chart-bar",
        ORANGE,
        "Individual macro-F1: before / after\nWithin / across / excluded species\nVary calibration-set size",
    )
    ax.text(
        xs2[2] + 0.18,
        y + 0.28,
        "Assess alignment and classification",
        fontsize=10,
        color=ORANGE,
        fontweight="bold",
    )
    for x, next_x in zip(xs2[:-1], xs2[1:]):
        arrow((x + w + 0.06, y + h / 2), (next_x - 0.08, y + h / 2), ORANGE)
    # State the shared frozen-model condition without implying the exclusion model is transferred.
    ax.text(
        14.65,
        3.80,
        "Original classifier held fixed",
        ha="right",
        va="center",
        fontsize=10,
        color=BLUE,
        bbox=dict(boxstyle="round,pad=.35", facecolor="#edf4f8", edgecolor="none"),
    )
    ax.text(
        0.20,
        0.21,
        "Icons are schematic; they do not depict measured equipment geometry.",
        fontsize=9,
        color=muted,
    )
    ax.text(
        14.80,
        0.21,
        "SVG icons: Tabler Icons · MIT licence",
        ha="right",
        fontsize=9,
        color=muted,
    )
    fig._mee_manual_layout = True
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(OUT / "figures/figure1_workflow.svg", bbox_inches="tight")
    return fig


def visual_transform(tensor, kind, level=0):
    sys.path.insert(0, str(ROOT / "src"))
    from worm_species.data.conditions import (
        BinaryForegroundMask,
        ColourRetention,
        GaussianBlurPercent,
        ResolutionLoss,
        PatchShuffle,
    )

    if kind == "silhouette":
        return BinaryForegroundMask()(tensor)
    if kind == "grey":
        return ColourRetention(0)(tensor)
    if kind == "blur":
        return GaussianBlurPercent(level, max_sigma=64)(tensor)
    if kind == "resolution":
        return ResolutionLoss(level)(tensor)
    if kind == "patch":
        return PatchShuffle(int(level), seed=2026)(tensor)
    if kind == "blur_grey":
        return ColourRetention(0)(GaussianBlurPercent(level, max_sigma=64)(tensor))
    if kind == "blur_patch":
        return PatchShuffle(16, seed=2026)(
            GaussianBlurPercent(level, max_sigma=64)(tensor)
        )
    return tensor
