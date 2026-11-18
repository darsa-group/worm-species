"""Plan or explicitly execute the full training-to-analysis workflow."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scripts import paper_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run"])
    parser.add_argument("--config", type=Path, default=ROOT / "work/reproduction.yaml")
    parser.add_argument(
        "--training-config", type=Path, default=ROOT / "configs/train.yaml"
    )
    parser.add_argument("--device")
    args = parser.parse_args()
    cfg, train, root = paper_training.settings(args.config, args.training_config)
    if args.device:
        train["device"] = args.device
    plan = paper_training.prepare(cfg, train, root)
    print(
        {
            "fits": len(plan["jobs"]),
            "counts": plan["counts"],
            "analysis_config": plan["analysis_config"],
        },
        flush=True,
    )
    if args.command == "plan":
        return
    # Keep fits sequential so one run does not compete with another for GPU memory.
    for job in plan["jobs"]:
        paper_training.run_job(job)
    config = plan["analysis_config"]
    # Later stages read the checkpoints and files written by the earlier stages.
    for command in [
        "plan",
        "inference",
        "features",
        "analysis",
        "adaptive",
        "controls-compile",
        "supplement",
        "figures",
        "verify",
    ]:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/paper.py"),
                command,
                "--config",
                config,
            ],
            cwd=ROOT,
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/publication_figures.py"),
            "--config",
            config,
            "--output",
            str(root / "publication"),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
