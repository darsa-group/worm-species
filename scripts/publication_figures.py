"""Build the five main publication layouts from reproduced result tables."""

from pathlib import Path
import argparse
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work/matplotlib-cache"))
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import pandas as pd
from scripts import (
    paper,
    publication_layouts as layout,
    publication_examples as examples,
)
from scripts import publication_control_layout as controls


def build(cfg, output):
    output = Path(output).resolve()
    analysis = Path(cfg["paths"]["analysis"])
    if output.exists():
        raise FileExistsError(
            "Preserve existing publication plots; choose a fresh output directory"
        )
    for name in [
        "visual_task_summary.csv",
        "baseline_summary.csv",
        "global_calibration_sampling_summary.csv",
        "calibration_summary.csv",
        "chance_controls.csv",
        "domain_differences.csv",
        "matched_stage_bootstrap_summary.csv",
    ]:
        if not (analysis / "tables" / name).is_file():
            raise FileNotFoundError(f"Complete the saved-result analysis first: {name}")
    output.mkdir(parents=True)
    (output / "figures").mkdir()
    support = output / "supporting_data"
    support.mkdir()
    for path in (analysis / "tables").glob("*.csv"):
        shutil.copyfile(path, support / path.name)
    counts = pd.read_csv(Path(cfg["paths"]["controls"]) / "training/conditions.csv")
    counts[counts.kind.eq("stage_removed")].to_csv(
        support / "matched_removed_counts.csv", index=False
    )
    rows = pd.read_csv(
        Path(cfg["paths"]["dataset"]) / "metadata/images.csv", keep_default_na=False
    )
    chosen = json.loads((ROOT / "configs/figure_examples.json").read_text())
    for name, ids in chosen.items():
        sample = rows.set_index("image_id", drop=False).loc[ids].copy()
        sample["source_segmented"] = ""
        sample.to_csv(
            support
            / (
                "figure2_example_source.csv"
                if name == "visual"
                else "figure4_example_sources.csv"
            ),
            index=False,
        )
    shutil.copytree(ROOT / "assets", output / "assets")
    layout.OUT = output
    layout.DATA = analysis / "tables"
    layout.ANALYSIS = analysis
    controls.OUT = output
    controls.TABLE = support
    examples.OUT = output
    examples.DATA = Path(cfg["paths"]["dataset"])
    layout.plt.rcParams.update(
        {
            "font.size": 10,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    layout.savefig(examples.make_figure1(), "figure1_workflow")
    layout.savefig(layout.visual_figure(examples), "figure2_visual_information")
    layout.savefig(controls.figure(), "figure3_biological_coverage")
    layout.savefig(layout.external_figure(examples), "figure4_external_domains")
    layout.savefig(layout.calibration_figure(), "figure5_calibration")
    (output / "figure_receipt.json").write_text(
        json.dumps(
            {
                "main_figures": 5,
                "formats": ["PDF", "PNG"],
                "data": "configured reproduced result tables",
                "human_observation_records_generated": False,
                "new_training": False,
                "new_performance_estimates": False,
            },
            indent=2,
        )
        + "\n"
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "work/new_training/analysis.yaml"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "work/publication")
    args = parser.parse_args()
    print(build(paper.load_settings(args.config), args.output))
