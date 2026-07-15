"""Optional Streamlit entry point for the read-only result dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Streamlit executes the file path directly, so make the repository root
# importable without requiring an editable package installation.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.index import default_cache_path, load_index, refresh_index
from dashboard.views import render_dashboard


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results-root", type=Path, default=Path("outputs_slurm"))
    parser.add_argument("--cache", type=Path, default=None)
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
    st.set_page_config(page_title="Worm species experiments", layout="wide")
    st.title("Worm species experiment dashboard")
    st.caption("Read-only view. Status is inferred from files; the SLURM scheduler is not queried.")

    @st.cache_data(ttl=60, show_spinner=False)
    def cached_refresh(results_root: str, cache_path: str, max_depth: int) -> dict:
        return refresh_index(results_root, cache_path, max_depth=max_depth)

    if st.sidebar.button("Refresh index"):
        cached_refresh.clear()
    with st.spinner("Scanning lightweight result metadata…"):
        summary = cached_refresh(str(args.results_root), str(cache), args.max_depth)
    st.sidebar.caption(f"Indexed {summary['runs']} runs")
    try:
        index = load_index(cache)
    except Exception as exc:
        st.error(f"Could not load dashboard index: {exc}")
        return 1
    st.sidebar.caption(f"Results: {args.results_root.absolute()}")
    st.sidebar.caption(f"Index: {cache.absolute()}")
    render_dashboard(st, index["experiments"], index["runs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
