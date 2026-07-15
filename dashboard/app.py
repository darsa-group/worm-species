"""Optional Streamlit entry point for the read-only result dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Streamlit executes the file path directly, so make the repository root
# importable without requiring an editable package installation.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.index import (
    default_cache_path,
    default_sources,
    load_derived_records,
    load_index,
    parse_source,
    refresh_indexes,
)
from dashboard.views import render_dashboard


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--results-root", action="append", default=[])
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--derived-cache", type=Path, default=None)
    parser.add_argument("--max-depth", type=int, default=8)
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        import streamlit as st
    except ImportError:
        print(
            "Streamlit is not installed. Install dashboard/requirements.txt, then run:\n"
            "streamlit run dashboard/app.py -- --results-root outputs_slurm",
            file=sys.stderr,
        )
        return 2

    args = _arguments(argv)
    cache = args.cache or default_cache_path()
    repository_root = Path(__file__).resolve().parents[1]
    raw_sources = [*args.source, *args.results_root]
    try:
        sources = (
            [parse_source(value) for value in raw_sources]
            if raw_sources
            else default_sources(repository_root)
        )
    except ValueError as exc:
        st.error(f"Invalid result source: {exc}")
        return 2
    derived_cache = args.derived_cache or cache.parent / "derived"
    st.set_page_config(page_title="Worm species experiments", layout="wide")
    st.title("Worm species experiment dashboard")
    st.caption("Read-only view. Status is inferred from files; the SLURM scheduler is not queried.")

    @st.cache_data(ttl=60, show_spinner=False)
    def cached_refresh(source_values: tuple[str, ...], cache_path: str, max_depth: int) -> dict:
        return refresh_indexes(
            [parse_source(value) for value in source_values],
            cache_path,
            max_depth=max_depth,
        )

    if st.sidebar.button("Refresh index"):
        cached_refresh.clear()
    with st.spinner("Scanning lightweight result metadata…"):
        summary = cached_refresh(
            tuple(f"{source.label}={source.path}" for source in sources),
            str(cache),
            args.max_depth,
        )
    st.sidebar.caption(f"Indexed {summary['runs']} runs")
    try:
        index = load_index(cache)
    except Exception as exc:
        st.error(f"Could not load dashboard index: {exc}")
        return 1
    for source in sources:
        st.sidebar.caption(f"{source.label}: {source.path}")
    st.sidebar.caption(f"Index: {cache.absolute()}")
    derived_records, derived_warnings = load_derived_records(derived_cache)
    if derived_warnings:
        st.sidebar.caption(f"Derived cache: {len(derived_warnings)} warning(s)")
    elif derived_records:
        st.sidebar.caption(f"Derived cache: {len(derived_records)} run summaries")
    else:
        st.sidebar.caption("Derived cache: not prepared")
    render_dashboard(
        st,
        index["experiments"],
        index["runs"],
        derived_records=derived_records,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
