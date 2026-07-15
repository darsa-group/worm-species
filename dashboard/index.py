"""SQLite-backed incremental cache for dashboard discovery records."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .data_loader import discover_results


SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SourceSpec:
    """A labelled result root kept distinct in the shared index."""

    label: str
    path: Path
    kind: str


def _source_kind(label: str, path: Path) -> str:
    if label in {"slurm", "single_task"}:
        return label
    if path.name == "outputs_slurm" or "outputs_slurm" in path.parts:
        return "slurm"
    if "single_task" in path.parts:
        return "single_task"
    return "local"


def parse_source(value: str) -> SourceSpec:
    """Parse ``LABEL=PATH`` while retaining path-only CLI compatibility."""

    if "=" in value:
        label, raw_path = value.split("=", 1)
        if not label.strip() or not raw_path.strip():
            raise ValueError(f"invalid labelled result source: {value!r}")
    else:
        raw_path = value
        candidate = Path(raw_path)
        if candidate.name == "outputs_slurm" or "outputs_slurm" in candidate.parts:
            label = "slurm"
        elif "single_task" in candidate.parts:
            label = "single_task"
        else:
            label = candidate.name or "results"
    path = Path(raw_path).expanduser().absolute()
    return SourceSpec(label=label.strip(), path=path, kind=_source_kind(label.strip(), path))


def default_sources(repository_root: str | Path | None = None) -> list[SourceSpec]:
    root = Path(repository_root or Path.cwd()).expanduser().absolute()
    candidates = [
        SourceSpec("slurm", root / "outputs_slurm", "slurm"),
        SourceSpec("single_task", root / "single_task" / "outputs", "single_task"),
    ]
    existing = [source for source in candidates if source.path.is_dir()]
    return existing or candidates[:1]


def _normalise_sources(sources: Iterable[SourceSpec | str | Path]) -> list[SourceSpec]:
    result: list[SourceSpec] = []
    labels: set[str] = set()
    paths: set[Path] = set()
    for source in sources:
        spec = source if isinstance(source, SourceSpec) else parse_source(str(source))
        path = spec.path.expanduser().absolute()
        spec = SourceSpec(spec.label, path, spec.kind)
        if spec.label in labels:
            raise ValueError(f"duplicate result source label: {spec.label}")
        if path in paths:
            raise ValueError(f"duplicate result source path: {path}")
        labels.add(spec.label)
        paths.add(path)
        result.append(spec)
    if not result:
        raise ValueError("at least one result source is required")
    return result


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def default_cache_path() -> Path:
    repository_root = Path(__file__).resolve().parents[1]
    return repository_root / ".cache" / "worm-species-dashboard" / "index.sqlite3"


def validate_cache_path(
    cache_path: str | Path,
    results_roots: str | Path | Sequence[str | Path],
) -> Path:
    cache = Path(cache_path).expanduser().absolute()
    resolved_cache = cache.resolve(strict=False)
    roots = (
        [results_roots]
        if isinstance(results_roots, (str, Path))
        else list(results_roots)
    )
    for results_root in roots:
        root = Path(results_root).expanduser().resolve()
        if _is_relative_to(resolved_cache, root):
            raise ValueError(f"dashboard cache must be outside results root: {cache}")
    return cache


def _connect(path: Path, *, rebuild_incompatible: bool = False) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    metadata_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
    ).fetchone()
    if metadata_exists:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        if row and int(row[0]) != SCHEMA_VERSION:
            if not rebuild_incompatible:
                connection.close()
                raise ValueError(
                    "dashboard index schema is outdated; refresh the index to rebuild it"
                )
            connection.executescript(
                "DROP TABLE IF EXISTS runs; DROP TABLE IF EXISTS experiments; "
                "DROP TABLE IF EXISTS metadata;"
            )
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS experiments (
            uid TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            updated_at REAL NOT NULL,
            signature TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            scan_generation INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            uid TEXT PRIMARY KEY,
            experiment_uid TEXT NOT NULL,
            run_name TEXT NOT NULL,
            path TEXT NOT NULL,
            status TEXT NOT NULL,
            model TEXT,
            updated_at REAL NOT NULL,
            signature TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            scan_generation INTEGER NOT NULL,
            FOREIGN KEY (experiment_uid) REFERENCES experiments(uid) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS runs_experiment_idx ON runs(experiment_uid);
        CREATE INDEX IF NOT EXISTS runs_status_idx ON runs(status);
        CREATE INDEX IF NOT EXISTS runs_model_idx ON runs(model);
        """
    )
    return connection


def _next_generation(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT value FROM metadata WHERE key='scan_generation'").fetchone()
    return int(row[0]) + 1 if row else 1


def refresh_indexes(
    sources: Sequence[SourceSpec | str | Path],
    cache_path: str | Path | None = None,
    *,
    max_depth: int = 8,
) -> dict[str, Any]:
    """Atomically refresh one cache generation from all labelled roots."""

    selected_sources = _normalise_sources(sources)
    cache = validate_cache_path(
        cache_path or default_cache_path(), [source.path for source in selected_sources]
    )
    discoveries = [
        discover_results(
            source.path,
            max_depth=max_depth,
            source_kind=source.kind,
            source_label=source.label,
        )
        for source in selected_sources
    ]
    cache.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect(cache, rebuild_incompatible=True)
    try:
        with connection:
            generation = _next_generation(connection)
            for discovery in discoveries:
                for experiment in discovery.experiments:
                    payload = experiment.to_dict()
                    connection.execute(
                        """
                        INSERT INTO experiments
                            (uid, name, path, updated_at, signature, payload_json, scan_generation)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(uid) DO UPDATE SET
                            name=excluded.name, path=excluded.path,
                            updated_at=excluded.updated_at, signature=excluded.signature,
                            payload_json=excluded.payload_json,
                            scan_generation=excluded.scan_generation
                        """,
                        (
                            experiment.uid,
                            experiment.name,
                            experiment.path,
                            experiment.updated_at,
                            experiment.signature,
                            json.dumps(payload, sort_keys=True),
                            generation,
                        ),
                    )
                for run in discovery.runs:
                    payload = run.to_dict()
                    connection.execute(
                        """
                        INSERT INTO runs
                            (uid, experiment_uid, run_name, path, status, model,
                             updated_at, signature, payload_json, scan_generation)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(uid) DO UPDATE SET
                            experiment_uid=excluded.experiment_uid,
                            run_name=excluded.run_name, path=excluded.path,
                            status=excluded.status, model=excluded.model,
                            updated_at=excluded.updated_at, signature=excluded.signature,
                            payload_json=excluded.payload_json,
                            scan_generation=excluded.scan_generation
                        """,
                        (
                            run.uid,
                            run.experiment_uid,
                            run.run_name,
                            run.path,
                            run.status.value,
                            run.model,
                            run.updated_at,
                            run.signature,
                            json.dumps(payload, sort_keys=True),
                            generation,
                        ),
                    )
            connection.execute("DELETE FROM runs WHERE scan_generation != ?", (generation,))
            connection.execute("DELETE FROM experiments WHERE scan_generation != ?", (generation,))
            metadata = {
                "schema_version": str(SCHEMA_VERSION),
                "scan_generation": str(generation),
                "results_roots": json.dumps(
                    [
                        {"label": source.label, "kind": source.kind, "path": str(source.path)}
                        for source in selected_sources
                    ],
                    sort_keys=True,
                ),
                "refreshed_at": str(time.time()),
                "discovery_warnings": json.dumps(
                    [
                        {"source_label": source.label, **warning.__dict__}
                        for source, discovery in zip(selected_sources, discoveries)
                        for warning in discovery.warnings
                    ],
                    sort_keys=True,
                ),
            }
            connection.executemany(
                """
                INSERT INTO metadata(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                metadata.items(),
            )
    finally:
        connection.close()
    return {
        "cache_path": str(cache),
        "sources": [
            {"label": source.label, "kind": source.kind, "path": str(source.path)}
            for source in selected_sources
        ],
        "experiments": sum(len(discovery.experiments) for discovery in discoveries),
        "runs": sum(len(discovery.runs) for discovery in discoveries),
        "warnings": sum(
            len(discovery.warnings) + sum(len(run.warnings) for run in discovery.runs)
            for discovery in discoveries
        ),
    }


def refresh_index(
    results_root: str | Path,
    cache_path: str | Path | None = None,
    *,
    max_depth: int = 8,
) -> dict[str, Any]:
    """Backward-compatible single-root index refresh."""

    return refresh_indexes([parse_source(str(results_root))], cache_path, max_depth=max_depth)


def load_index(cache_path: str | Path | None = None) -> dict[str, Any]:
    cache = Path(cache_path or default_cache_path()).expanduser().absolute()
    if not cache.is_file():
        raise FileNotFoundError(f"dashboard index does not exist: {cache}")
    connection = _connect(cache)
    try:
        metadata = {
            row["key"]: row["value"]
            for row in connection.execute("SELECT key, value FROM metadata")
        }
        experiments = [
            json.loads(row["payload_json"])
            for row in connection.execute("SELECT payload_json FROM experiments ORDER BY name")
        ]
        runs = [
            json.loads(row["payload_json"])
            for row in connection.execute(
                "SELECT payload_json FROM runs ORDER BY updated_at DESC, run_name"
            )
        ]
    finally:
        connection.close()
    return {"metadata": metadata, "experiments": experiments, "runs": runs}


def _safe_cache_member(cache_root: Path, relative_path: str) -> Path:
    candidate = (cache_root / relative_path).resolve(strict=False)
    root = cache_root.resolve(strict=False)
    if not _is_relative_to(candidate, root):
        raise ValueError(f"derived cache path escapes cache root: {relative_path}")
    return candidate


def load_derived_records(
    cache_root: str | Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    """Load bounded derived summaries, never following manifest paths outside cache."""

    root = Path(cache_root).expanduser().absolute()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {}, []
    warnings: list[str] = []
    try:
        if manifest_path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("derived manifest exceeds 20 MiB")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, [f"Could not read derived manifest: {exc}"]
    if manifest.get("schema_version") != 1:
        return {}, [f"Unsupported derived manifest schema: {manifest.get('schema_version')!r}"]
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for item in manifest.get("runs", []):
        if not isinstance(item, dict):
            warnings.append("Ignored malformed derived run entry")
            continue
        label = item.get("source_label")
        uid = item.get("run_uid")
        relative_summary = item.get("summary")
        if not all(isinstance(value, str) and value for value in (label, uid, relative_summary)):
            warnings.append("Ignored derived run entry without source_label, run_uid, or summary")
            continue
        try:
            summary_path = _safe_cache_member(root, relative_summary)
            if summary_path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("summary exceeds 2 MiB")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            image_relative = summary.get("combined_confusion_matrix_image")
            if image_relative:
                image_path = _safe_cache_member(root, str(image_relative))
                summary["combined_confusion_matrix_image_path"] = (
                    str(image_path) if image_path.is_file() else None
                )
            records[(label, uid)] = summary
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"Could not read derived summary for {label}/{uid}: {exc}")
    return records, warnings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the read-only worm-species result index")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Labelled result root; repeat to combine SLURM and local results",
    )
    parser.add_argument(
        "--results-root",
        action="append",
        default=[],
        metavar="PATH",
        help="Backward-compatible result root; repeat to combine roots",
    )
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="Print summary as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        raw_sources = [*args.source, *args.results_root]
        sources = [parse_source(value) for value in raw_sources] if raw_sources else default_sources()
        summary = refresh_indexes(sources, args.cache, max_depth=args.max_depth)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"Indexed {summary['runs']} runs in {summary['experiments']} experiments "
            f"from {len(summary['sources'])} sources to {summary['cache_path']} "
            f"({summary['warnings']} warnings)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
