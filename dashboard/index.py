"""SQLite-backed incremental cache for dashboard discovery records."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .data_loader import discover_results
SCHEMA_VERSION = 1


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def default_cache_path() -> Path:
    repository_root = Path(__file__).resolve().parents[1]
    return repository_root / ".cache" / "worm-species-dashboard" / "index.sqlite3"


def validate_cache_path(cache_path: str | Path, results_root: str | Path) -> Path:
    cache = Path(cache_path).expanduser().absolute()
    root = Path(results_root).expanduser().resolve()
    resolved_cache = cache.resolve(strict=False)
    if _is_relative_to(resolved_cache, root):
        raise ValueError(f"dashboard cache must be outside results root: {cache}")
    return cache


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
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


def refresh_index(
    results_root: str | Path,
    cache_path: str | Path | None = None,
    *,
    max_depth: int = 8,
) -> dict[str, Any]:
    """Rescan lightweight metadata and atomically refresh the external cache."""

    root = Path(results_root).expanduser().absolute()
    cache = validate_cache_path(cache_path or default_cache_path(), root)
    discovery = discover_results(root, max_depth=max_depth)
    cache.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect(cache)
    try:
        with connection:
            generation = _next_generation(connection)
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
                "results_root": discovery.results_root,
                "refreshed_at": str(time.time()),
                "discovery_warnings": json.dumps(
                    [warning.__dict__ for warning in discovery.warnings], sort_keys=True
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
        "results_root": discovery.results_root,
        "experiments": len(discovery.experiments),
        "runs": len(discovery.runs),
        "warnings": len(discovery.warnings) + sum(len(run.warnings) for run in discovery.runs),
    }


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the read-only worm-species result index")
    parser.add_argument("--results-root", default="outputs_slurm")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="Print summary as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        summary = refresh_index(args.results_root, args.cache, max_depth=args.max_depth)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"Indexed {summary['runs']} runs in {summary['experiments']} experiments "
            f"to {summary['cache_path']} ({summary['warnings']} warnings)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
