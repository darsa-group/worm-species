#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1

set -euo pipefail

: "${RESULTS_ROOT:?}"
: "${CONDA_SH:?}"
: "${CONDA_ENV:?}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

python - "$RESULTS_ROOT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
rows = []

for summary_path in sorted(root.rglob("run_summary.json")):
    try:
        row = json.loads(summary_path.read_text())
    except Exception as exc:
        print(f"Skipping unreadable summary {summary_path}: {exc}")
        continue

    row["summary_path"] = str(summary_path.relative_to(root))
    rows.append(row)

if rows:
    df = pd.DataFrame(rows)
    if "colour_percent" in df.columns:
        df = df.sort_values("colour_percent", ascending=False).reset_index(drop=True)
    out_path = root / "colour_ablation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} completed runs to {out_path}")
else:
    print("No run_summary.json files were found.")

failed = []
for status_path in sorted(root.glob("*/run_status.txt")):
    status = status_path.read_text().strip()
    if status != "0":
        failed.append({
            "run_name": status_path.parent.name,
            "status": status,
        })

if failed:
    failed_path = root / "failed_runs.csv"
    pd.DataFrame(failed).to_csv(failed_path, index=False)
    print(f"Recorded {len(failed)} failed runs in {failed_path}")
PY
