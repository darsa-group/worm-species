#!/usr/bin/env python3
"""Recover complete target-class metrics from retained full-test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.worm_species.training.metrics import classification_metric_summary


STAGES = {
    "adult_taxon_baseline": "data_holdout_control_evaluation",
    "adult_taxon_holdouts": "data_holdout_evaluation",
}


def _probability_column(
    predictions: pd.DataFrame,
    target_label: str,
) -> np.ndarray | None:
    if "class_probabilities_json" not in predictions:
        return None
    values = []
    try:
        for payload in predictions["class_probabilities_json"]:
            mapping = json.loads(str(payload))
            values.append(float(mapping[target_label]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return np.asarray(values, dtype=float)


def augment_run(run_dir: Path, evaluation_directory: str) -> dict:
    metrics_path = run_dir / evaluation_directory / "task_metrics.csv"
    predictions_path = run_dir / "test_predictions_best.csv"
    output_path = (
        run_dir / evaluation_directory / "target_class_metrics_full_test.csv"
    )
    if not metrics_path.is_file():
        return {"run_dir": str(run_dir), "status": "missing_task_metrics"}
    if not predictions_path.is_file():
        return {"run_dir": str(run_dir), "status": "missing_test_predictions"}
    rows = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    output_rows = []
    for row in rows.itertuples(index=False):
        task = str(row.task)
        target_label = getattr(row, "target_label", None)
        task_predictions = predictions[predictions["task"].astype(str).eq(task)]
        record = {
            "holdout": getattr(row, "holdout", None),
            "cohort": getattr(row, "cohort", None),
            "task": task,
            "target_label": target_label,
            "metric_scope": "complete independent test split",
            "metric_unit": "image",
        }
        if task_predictions.empty or pd.isna(target_label):
            record["status"] = "missing_task_predictions_or_target"
            output_rows.append(record)
            continue
        labels = list(dict.fromkeys([
            *task_predictions["true_label"].astype(str).tolist(),
            *task_predictions["predicted_label"].astype(str).tolist(),
        ]))
        target_label = str(target_label)
        if target_label not in labels:
            record["status"] = "target_not_supported"
            output_rows.append(record)
            continue
        label_to_index = {label: index for index, label in enumerate(labels)}
        probabilities = _probability_column(task_predictions, target_label)
        summary = classification_metric_summary(
            task_predictions["true_label"].astype(str).map(label_to_index).to_numpy(),
            task_predictions["predicted_label"].astype(str).map(label_to_index).to_numpy(),
            target_index=label_to_index[target_label],
            target_probabilities=probabilities,
        )
        record.update(summary)
        record["status"] = "complete"
        record["probability_metrics_available"] = probabilities is not None
        output_rows.append(record)
    output = pd.DataFrame(output_rows)
    output.to_csv(output_path, index=False)
    return {
        "run_dir": str(run_dir),
        "status": "complete",
        "rows": int(len(output)),
        "probability_rows": int(
            output.get("probability_metrics_available", pd.Series(dtype=bool))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "output": str(output_path),
    }


def augment_results(root: Path) -> dict:
    records = []
    for stage, evaluation_directory in STAGES.items():
        for metrics_path in sorted(
            (root / "runs" / stage).rglob(
                f"{evaluation_directory}/task_metrics.csv"
            )
        ):
            records.append(
                augment_run(metrics_path.parents[1], evaluation_directory)
            )
    summary = {
        "result_root": str(root),
        "runs_seen": len(records),
        "status_counts": pd.Series(
            [record["status"] for record in records], dtype=str
        ).value_counts().to_dict(),
        "records": records,
        "retraining_required": False,
        "evaluation_required": [
            record["run_dir"]
            for record in records
            if record["status"] == "missing_test_predictions"
        ],
        "note": (
            "Missing hard predictions require checkpoint re-evaluation, not "
            "retraining, when best_model.pt is retained. Probability metrics "
            "require predictions written by the enhanced evaluator."
        ),
    }
    output = root / "data_ablation_metric_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root", type=Path, default=Path("publication_30seed_result")
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail if any completed metric table lacks full-test predictions.",
    )
    args = parser.parse_args()
    summary = augment_results(args.result_root)
    print(json.dumps({
        "runs_seen": summary["runs_seen"],
        "status_counts": summary["status_counts"],
        "evaluation_required": len(summary["evaluation_required"]),
    }, indent=2, sort_keys=True))
    if args.strict and summary["evaluation_required"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
