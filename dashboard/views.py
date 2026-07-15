"""Streamlit rendering functions kept separate from result discovery."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .data_loader import load_csv_rows


def _artifact(run: dict[str, Any], kind: str) -> dict[str, Any] | None:
    return next((item for item in run.get("artifacts", []) if item.get("kind") == kind), None)


def _metric(run: dict[str, Any], metric: str, task: str | None = None) -> float | None:
    for item in run.get("metrics", []):
        if item.get("metric") == metric and item.get("task") == task:
            return item.get("value")
    return None


def _display_value(value: Any) -> Any:
    return "—" if value is None else value


def _facet(run: dict[str, Any], key: str) -> Any:
    return run.get("hyperparameters", {}).get(key)


def _option(value: Any) -> str:
    if value is None:
        return "(unspecified)"
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    return str(value)


def _available_table(artifact: dict[str, Any] | None, max_rows: int = 5_000) -> list[dict[str, str]]:
    if not artifact or not artifact.get("available"):
        return []
    try:
        return load_csv_rows(artifact["path"], max_rows=max_rows)
    except Exception:
        return []


def _filter_runs(st: Any, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not runs:
        return []
    experiments = sorted({item["experiment_name"] for item in runs})
    sources = sorted({item.get("source_label", "unknown") for item in runs})
    unknown_model = "(unknown)"
    models = sorted({item["model"] or unknown_model for item in runs})
    statuses = sorted({item["status"] for item in runs})
    conditions = sorted({item["train_condition"] for item in runs if item.get("train_condition")})
    features = sorted({item["train_feature"] for item in runs if item.get("train_feature")})
    transforms = sorted({item["train_transform"] for item in runs if item.get("train_transform")})
    tasks = sorted({task for item in runs for task in item.get("tasks", [])})
    dates = [datetime.fromtimestamp(item["updated_at"]).date() for item in runs]
    with st.sidebar:
        st.header("Filters")
        selected_sources = st.multiselect("Result source", sources, default=sources)
        selected_experiments = st.multiselect("Experiment", experiments, default=experiments)
        selected_models = st.multiselect("Architecture", models, default=models)
        selected_statuses = st.multiselect("Status", statuses, default=statuses)
        selected_conditions = st.multiselect("Training condition", conditions, default=conditions)
        selected_features = st.multiselect("Condition feature", features, default=features)
        selected_transforms = st.multiselect("Condition transform", transforms, default=transforms)
        selected_tasks = st.multiselect("Task", tasks, default=tasks)
        facet_selections: dict[str, set[str]] = {}
        for key, label in (
            ("epochs", "Configured epochs"),
            ("learning_rate", "Learning rate"),
            ("weight_decay", "Weight decay"),
            ("batch_size", "Batch size"),
            ("pretrained", "Pretrained weights"),
            ("freeze_backbone", "Frozen backbone"),
            ("class_weight", "Class-weighted loss"),
            ("age_loss_weight", "Age loss weight"),
            ("species_loss_weight", "Species loss weight"),
            ("genus_loss_weight", "Genus loss weight"),
            ("hierarchy_loss_enabled", "Hierarchy loss"),
            ("wandb_enabled", "W&B logging"),
        ):
            values = sorted({_option(_facet(item, key)) for item in runs})
            facet_selections[key] = set(st.multiselect(label, values, default=values))
        start_date = st.date_input("Updated on or after", min(dates))
        end_date = st.date_input("Updated on or before", max(dates))
    return [
        item for item in runs
        if item.get("source_label", "unknown") in selected_sources
        and item["experiment_name"] in selected_experiments
        and (item.get("model") or unknown_model) in selected_models
        and item["status"] in selected_statuses
        and (not item.get("train_condition") or item.get("train_condition") in selected_conditions)
        and (not item.get("train_feature") or item.get("train_feature") in selected_features)
        and (not item.get("train_transform") or item.get("train_transform") in selected_transforms)
        and (not item.get("tasks") or not selected_tasks or set(item.get("tasks", [])) & set(selected_tasks))
        and all(_option(_facet(item, key)) in selected for key, selected in facet_selections.items())
        and start_date <= datetime.fromtimestamp(item["updated_at"]).date() <= end_date
    ]


def _overview(st: Any, runs: list[dict[str, Any]], experiments: list[dict[str, Any]]) -> None:
    columns = st.columns(4)
    columns[0].metric("Experiments", len({item["experiment_uid"] for item in runs}))
    columns[1].metric("Runs", len(runs))
    columns[2].metric("Completed", sum(item["status"] == "completed" for item in runs))
    columns[3].metric(
        "Needs attention",
        sum(item["status"] in {"failed", "incomplete", "possibly_active"} for item in runs),
    )
    status_rows = [
        {
            "source": experiment.get("source_label"),
            "experiment": experiment["name"],
            "expected_runs": experiment.get("expected_run_count"),
            "indexed_runs": sum(run["experiment_uid"] == experiment["uid"] for run in runs),
            "path": experiment["path"],
        }
        for experiment in experiments
        if any(run["experiment_uid"] == experiment["uid"] for run in runs)
    ]
    if status_rows:
        st.dataframe(status_rows, use_container_width=True, hide_index=True)

    comparison_rows = []
    for run in runs:
        facets = run.get("hyperparameters", {})
        comparison_rows.append(
            {
                "source": run.get("source_label"),
                "experiment": run.get("experiment_name"),
                "run": run.get("run_name"),
                "status": run.get("status"),
                "model": run.get("model"),
                "tasks": ", ".join(run.get("tasks", [])),
                "macro_f1": run.get("effective_macro_f1"),
                "macro_f1_definition": run.get("effective_macro_f1_label"),
                "epochs": facets.get("epochs"),
                "lr": facets.get("learning_rate"),
                "weight_decay": facets.get("weight_decay"),
                "batch_size": facets.get("batch_size"),
                "pretrained": facets.get("pretrained"),
                "freeze_backbone": facets.get("freeze_backbone"),
                "class_weight": facets.get("class_weight"),
                "genus_weight": facets.get("genus_loss_weight"),
                "species_weight": facets.get("species_loss_weight"),
                "age_weight": facets.get("age_loss_weight"),
                "hierarchy_loss": facets.get("hierarchy_loss_enabled"),
                "condition": run.get("train_condition"),
                "condition_feature": run.get("train_feature"),
                "condition_transform": run.get("train_transform"),
                "condition_strength": run.get("train_strength"),
            }
        )
    if comparison_rows:
        with st.expander("Filtered run comparison", expanded=False):
            st.dataframe(comparison_rows, use_container_width=True, hide_index=True)


def _history_view(st: Any, run: dict[str, Any]) -> None:
    artifact = _artifact(run, "history.csv")
    rows = _available_table(artifact)
    if not rows:
        st.info("No readable history.csv for this run.")
        return
    try:
        import pandas as pd

        frame = pd.DataFrame(rows)
        for column in frame.columns:
            if column != "epoch":
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
        choices = [column for column in frame if column != "epoch" and frame[column].notna().any()]
        defaults = [column for column in ("train_loss", "val_loss", "val_mean_macro_f1") if column in choices]
        selected = st.multiselect("History series", choices, default=defaults)
        if selected:
            st.line_chart(frame.set_index("epoch")[selected])
    except Exception as exc:
        st.warning(f"History could not be plotted: {exc}")


def _confusion_frame(rows: list[dict[str, str]], *, normalise: bool) -> Any:
    import pandas as pd

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    label_column = next(
        (column for column in frame.columns if column in {"", "Unnamed: 0"}),
        frame.columns[0],
    )
    labels = frame.pop(label_column).astype(str)
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    frame.index = labels
    if normalise:
        row_totals = frame.sum(axis=1).replace(0, float("nan"))
        frame = frame.div(row_totals, axis=0).fillna(0.0)
    return frame


def _reports_view(
    st: Any,
    run: dict[str, Any],
    derived: dict[str, Any] | None = None,
) -> None:
    reports = [item for item in run.get("artifacts", []) if item["kind"] == "classification_report"]
    matrices = [item for item in run.get("artifacts", []) if item["kind"] == "confusion_matrix"]
    left, right = st.columns(2)
    with left:
        st.subheader("Classification report")
        if reports:
            selected = st.selectbox("Report", reports, format_func=lambda item: Path(item["path"]).name)
            rows = _available_table(selected)
            st.dataframe(rows, use_container_width=True, hide_index=True) if rows else st.warning("Report unavailable or malformed")
        else:
            st.info("No classification report found.")
    with right:
        st.subheader("Confusion matrix")
        combined_image = (derived or {}).get("combined_confusion_matrix_image_path")
        if combined_image:
            st.image(combined_image, caption="Prepared combined task confusion matrices")
        if matrices:
            selected = st.selectbox("Matrix", matrices, format_func=lambda item: Path(item["path"]).name)
            rows = _available_table(selected)
            if rows:
                normalise = st.toggle("Row-normalise matrix", value=True)
                try:
                    frame = _confusion_frame(rows, normalise=normalise)
                    st.dataframe(
                        frame.style.background_gradient(cmap="Blues", axis=None),
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.warning(f"Matrix could not be rendered: {exc}")
                    st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.warning("Matrix unavailable or malformed")
        else:
            st.info("No confusion matrix found.")


def _cue_view(st: Any, run: dict[str, Any], experiment: dict[str, Any] | None) -> None:
    cue_artifacts = [item for item in run.get("artifacts", []) if item["kind"].startswith("cue_suppression/")]
    comparison = None
    if experiment:
        comparison = next(
            (item for item in experiment.get("artifacts", []) if item["kind"] == "matched_vs_rgb_stress_test.csv"),
            None,
        )
    if not cue_artifacts and not comparison:
        st.info("No cue-suppression or matched-vs-RGB artifacts found.")
        return
    if cue_artifacts:
        selected = st.selectbox("Cue-suppression table", cue_artifacts, format_func=lambda item: item["kind"])
        rows = _available_table(selected, max_rows=20_000)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
            if selected["kind"].endswith("macro_f1_ratios.csv"):
                try:
                    import pandas as pd

                    frame = pd.DataFrame(rows)
                    frame["strength"] = pd.to_numeric(frame.get("strength"), errors="coerce")
                    frame["ratio_to_original"] = pd.to_numeric(
                        frame.get("ratio_to_original"), errors="coerce"
                    )
                    tasks = sorted(frame["task"].dropna().unique())
                    task = st.selectbox("Cue ratio task", tasks) if tasks else None
                    curve = frame[frame["task"] == task] if task else frame
                    if curve["strength"].notna().any():
                        st.line_chart(
                            curve.sort_values("strength").set_index("strength")[["ratio_to_original"]]
                        )
                except Exception as exc:
                    st.warning(f"Cue-suppression curve could not be plotted: {exc}")
    if comparison:
        rows = _available_table(comparison, max_rows=100_000)
        matching = [row for row in rows if row.get("run_name") == run["run_name"]]
        st.subheader("Matched condition vs fixed-RGB stress")
        st.caption("Matched-condition training metrics and fixed-RGB stress metrics remain separate columns.")
        st.dataframe(matching or rows[:100], use_container_width=True, hide_index=True)
        try:
            import pandas as pd

            frame = pd.DataFrame(rows)
            saturation = frame[frame.get("train_transform") == "saturation"].copy()
            if not saturation.empty:
                saturation["train_strength"] = pd.to_numeric(
                    saturation["train_strength"], errors="coerce"
                )
                tasks = sorted(saturation["task"].dropna().unique())
                task = st.selectbox("Matched/RGB curve task", tasks) if tasks else None
                curve = saturation[saturation["task"] == task] if task else saturation
                numeric_columns = [
                    column for column in (
                        "matched_test_macro_f1",
                        "rgb_model_test_macro_f1",
                        "adaptation_gain_macro_f1",
                    )
                    if column in curve
                ]
                for column in numeric_columns:
                    curve[column] = pd.to_numeric(curve[column], errors="coerce")
                if numeric_columns:
                    st.line_chart(
                        curve.sort_values("train_strength").set_index("train_strength")[numeric_columns]
                    )
        except Exception as exc:
            st.warning(f"Matched/RGB curve could not be plotted: {exc}")


def _run_detail(
    st: Any,
    run: dict[str, Any],
    experiments_by_uid: dict[str, dict[str, Any]],
    derived: dict[str, Any] | None = None,
) -> None:
    st.header(run["run_name"])
    st.caption(
        f"{run.get('source_label', 'unknown')} / {run['status']} — {run['status_evidence']}"
    )
    derived_metrics = (derived or {}).get("metrics", {})
    if (derived or {}).get("run_type") == "multitask":
        macro_f1 = derived_metrics.get("effective_mean_macro_f1")
        macro_label = "Mean macro-F1 across tasks"
    elif (derived or {}).get("run_type") == "single_task":
        macro_f1 = derived_metrics.get("effective_task_macro_f1")
        macro_label = "Single-task macro-F1"
    else:
        macro_f1 = run.get("effective_macro_f1")
        macro_label = run.get("effective_macro_f1_label") or "Macro-F1"
    columns = st.columns(4)
    columns[0].metric("Best validation", _display_value(run.get("best_val_score")))
    columns[1].metric("Best epoch", _display_value(run.get("best_epoch")))
    columns[2].metric(macro_label, _display_value(macro_f1))
    columns[3].metric("Model", _display_value(run.get("model")))
    st.write(
        {
            "training_mode": run.get("training_mode"),
            "train_condition": run.get("train_condition"),
            "train_feature": run.get("train_feature"),
            "train_transform": run.get("train_transform"),
            "train_strength": run.get("train_strength"),
            "fixed_rgb_stress_evaluation": run.get("fixed_rgb_stress_evaluation"),
            "hyperparameters": run.get("hyperparameters", {}),
            "updated": datetime.fromtimestamp(run["updated_at"]).isoformat(timespec="seconds"),
        }
    )

    tabs = st.tabs(["Metrics", "History", "Reports", "Cue suppression", "Configuration", "Artifacts", "Warnings"])
    with tabs[0]:
        st.dataframe(run.get("metrics", []), use_container_width=True, hide_index=True)
    with tabs[1]:
        _history_view(st, run)
    with tabs[2]:
        _reports_view(st, run, derived)
    with tabs[3]:
        _cue_view(st, run, experiments_by_uid.get(run["experiment_uid"]))
    with tabs[4]:
        st.code(json.dumps(run.get("config", {}), indent=2, sort_keys=True), language="json")
        st.text_area("Applied overrides", run.get("overrides") or "", height=120)
    with tabs[5]:
        st.dataframe(run.get("artifacts", []), use_container_width=True, hide_index=True)
    with tabs[6]:
        warnings = [*run.get("warnings", []), *(derived or {}).get("warnings", [])]
        st.dataframe(warnings, use_container_width=True, hide_index=True) if warnings else st.success("No indexing warnings for this run.")


def render_dashboard(
    st: Any,
    experiments: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    *,
    derived_records: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> None:
    if not runs:
        st.info(
            "No experiment runs were indexed. Check the results-root path or "
            "refresh after result metadata is available."
        )
        return
    filtered = _filter_runs(st, runs)
    _overview(st, filtered, experiments)
    if not filtered:
        st.warning("No runs match the selected filters.")
        return
    choices = {
        f"{run.get('source_label', 'unknown')} / {run['experiment_name']} / "
        f"{run.get('array_run') or 'run'} / {run['run_name']}": run
        for run in filtered
    }
    selected_label = st.selectbox("Run", list(choices), index=0)
    selected_run = choices[selected_label]
    derived = (derived_records or {}).get(
        (selected_run.get("source_label", "unknown"), selected_run["uid"])
    )
    _run_detail(
        st,
        selected_run,
        {item["uid"]: item for item in experiments},
        derived,
    )
