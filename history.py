"""SQLite-backed evaluation history and lightweight HTML trend reporting."""
from __future__ import annotations

import html
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationSnapshot:
    run_id: str
    suite: str
    score: float
    pass_rate: float
    latency_ms: float | None
    cost_usd: float | None
    created_at: float


class EvaluationHistoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    suite TEXT NOT NULL,
                    score REAL NOT NULL,
                    pass_rate REAL NOT NULL,
                    latency_ms REAL,
                    cost_usd REAL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def record(
        self,
        *,
        run_id: str,
        suite: str,
        score: float,
        pass_rate: float,
        latency_ms: float | None = None,
        cost_usd: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not run_id.strip() or not suite.strip():
            raise ValueError("run_id and suite are required")
        if not 0 <= score <= 1 or not 0 <= pass_rate <= 1:
            raise ValueError("score and pass_rate must be between 0 and 1")
        if latency_ms is not None and latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if cost_usd is not None and cost_usd < 0:
            raise ValueError("cost_usd cannot be negative")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO evaluation_runs
                (run_id, suite, score, pass_rate, latency_ms, cost_usd, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    suite,
                    score,
                    pass_rate,
                    latency_ms,
                    cost_usd,
                    json.dumps(metadata or {}, sort_keys=True),
                    time.time(),
                ),
            )

    def recent(self, suite: str, limit: int = 20) -> list[EvaluationSnapshot]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, suite, score, pass_rate, latency_ms, cost_usd, created_at
                FROM evaluation_runs
                WHERE suite = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (suite, limit),
            ).fetchall()
        return [EvaluationSnapshot(*row) for row in rows]


def render_history_html(
    snapshots: list[EvaluationSnapshot], title: str = "Evaluation History"
) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(item.run_id)}</td>"
        f"<td>{html.escape(item.suite)}</td>"
        f"<td>{item.score:.3f}</td>"
        f"<td>{item.pass_rate:.3f}</td>"
        f"<td>{'' if item.latency_ms is None else f'{item.latency_ms:.1f}'}</td>"
        f"<td>{'' if item.cost_usd is None else f'{item.cost_usd:.4f}'}</td>"
        "</tr>"
        for item in snapshots
    )
    return (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        f"<h1>{html.escape(title)}</h1>"
        "<table><thead><tr><th>Run</th><th>Suite</th><th>Score</th>"
        "<th>Pass rate</th><th>Latency ms</th><th>Cost USD</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
