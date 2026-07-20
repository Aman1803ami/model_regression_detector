"""
SQLite persistence layer for evaluation results.

Stores eval run metadata, per-test-case results, and historical
score data for drift detection. Uses SQLite for zero-infrastructure
portability — the entire history travels with the repo.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from src.config import (
    CategoryMetrics,
    EmailCategory,
    EvalRunSummary,
    TestCaseResult,
    Difficulty,
)


# Default database path
_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "eval_results.db",
)


def _get_connection(db_path: str = _DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Get a SQLite connection, creating the database if needed."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str = _DEFAULT_DB_PATH) -> None:
    """
    Initialize the database schema.

    Creates tables if they don't exist. Safe to call multiple times.
    """
    conn = _get_connection(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                run_id TEXT PRIMARY KEY,
                prompt_version TEXT NOT NULL,
                model TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                total_cases INTEGER NOT NULL DEFAULT 0,
                passed_cases INTEGER NOT NULL DEFAULT 0,
                failed_cases INTEGER NOT NULL DEFAULT 0,
                error_cases INTEGER NOT NULL DEFAULT 0,
                overall_accuracy REAL NOT NULL DEFAULT 0.0,
                avg_summary_score REAL NOT NULL DEFAULT 0.0,
                avg_latency_ms REAL NOT NULL DEFAULT 0.0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                per_category_json TEXT DEFAULT '[]',
                is_baseline INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
                test_case_id TEXT NOT NULL,
                input_email TEXT NOT NULL,
                expected_category TEXT NOT NULL,
                expected_summary TEXT NOT NULL,
                predicted_category TEXT,
                predicted_summary TEXT DEFAULT '',
                category_match INTEGER NOT NULL DEFAULT 0,
                summary_relevance_score REAL NOT NULL DEFAULT 0.0,
                latency_ms REAL NOT NULL DEFAULT 0.0,
                tokens_used INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                difficulty TEXT DEFAULT 'medium',
                UNIQUE(run_id, test_case_id)
            );

            CREATE TABLE IF NOT EXISTS score_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
                timestamp TEXT NOT NULL,
                overall_accuracy REAL NOT NULL,
                avg_summary_score REAL NOT NULL,
                avg_latency_ms REAL NOT NULL,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                prompt_version TEXT NOT NULL,
                model TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_eval_results_run_id
                ON eval_results(run_id);
            CREATE INDEX IF NOT EXISTS idx_score_history_timestamp
                ON score_history(timestamp);
            CREATE INDEX IF NOT EXISTS idx_eval_runs_timestamp
                ON eval_runs(timestamp);
        """)
        conn.commit()
    finally:
        conn.close()


def save_run(
    summary: EvalRunSummary,
    db_path: str = _DEFAULT_DB_PATH,
    is_baseline: bool = False,
) -> str:
    """
    Persist an evaluation run and all its results to the database.

    Args:
        summary: Complete eval run summary with results.
        db_path: Path to the SQLite database.
        is_baseline: Whether to mark this as the baseline run.

    Returns:
        The run_id of the saved run.
    """
    init_db(db_path)
    conn = _get_connection(db_path)

    try:
        # Serialize per-category metrics
        per_category_json = json.dumps([
            {
                "category": cm.category.value,
                "total": cm.total,
                "correct": cm.correct,
                "accuracy": cm.accuracy,
                "avg_summary_score": cm.avg_summary_score,
                "avg_latency_ms": cm.avg_latency_ms,
            }
            for cm in summary.per_category
        ])

        # Insert run metadata
        conn.execute(
            """INSERT INTO eval_runs
               (run_id, prompt_version, model, timestamp, total_cases,
                passed_cases, failed_cases, error_cases, overall_accuracy,
                avg_summary_score, avg_latency_ms, total_tokens,
                per_category_json, is_baseline)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                summary.run_id,
                summary.prompt_version,
                summary.model,
                summary.timestamp.isoformat(),
                summary.total_cases,
                summary.passed_cases,
                summary.failed_cases,
                summary.error_cases,
                summary.overall_accuracy,
                summary.avg_summary_score,
                summary.avg_latency_ms,
                summary.total_tokens,
                per_category_json,
                1 if is_baseline else 0,
            ),
        )

        # Insert individual results
        for result in summary.results:
            conn.execute(
                """INSERT INTO eval_results
                   (run_id, test_case_id, input_email, expected_category,
                    expected_summary, predicted_category, predicted_summary,
                    category_match, summary_relevance_score, latency_ms,
                    tokens_used, error, difficulty)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    summary.run_id,
                    result.test_case_id,
                    result.input_email,
                    result.expected_category.value,
                    result.expected_summary,
                    result.predicted_category.value if result.predicted_category else None,
                    result.predicted_summary,
                    1 if result.category_match else 0,
                    result.summary_relevance_score,
                    result.latency_ms,
                    result.tokens_used,
                    result.error,
                    result.difficulty.value,
                ),
            )

        # Insert into score history for drift detection
        conn.execute(
            """INSERT INTO score_history
               (run_id, timestamp, overall_accuracy, avg_summary_score,
                avg_latency_ms, total_tokens, prompt_version, model)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                summary.run_id,
                summary.timestamp.isoformat(),
                summary.overall_accuracy,
                summary.avg_summary_score,
                summary.avg_latency_ms,
                summary.total_tokens,
                summary.prompt_version,
                summary.model,
            ),
        )

        conn.commit()
        return summary.run_id

    finally:
        conn.close()


def get_latest_run(db_path: str = _DEFAULT_DB_PATH) -> Optional[EvalRunSummary]:
    """Get the most recent evaluation run."""
    init_db(db_path)
    conn = _get_connection(db_path)

    try:
        row = conn.execute(
            "SELECT * FROM eval_runs ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()

        if row is None:
            return None

        return _row_to_summary(row, conn)
    finally:
        conn.close()


def get_baseline(db_path: str = _DEFAULT_DB_PATH) -> Optional[EvalRunSummary]:
    """
    Get the baseline run.

    Returns the run explicitly marked as baseline, or falls back
    to the second-most-recent run if no baseline is set.
    """
    init_db(db_path)
    conn = _get_connection(db_path)

    try:
        # Try to find an explicit baseline
        row = conn.execute(
            "SELECT * FROM eval_runs WHERE is_baseline = 1 ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()

        if row is not None:
            return _row_to_summary(row, conn)

        # Fall back to second most recent run
        rows = conn.execute(
            "SELECT * FROM eval_runs ORDER BY timestamp DESC LIMIT 2"
        ).fetchall()

        if len(rows) >= 2:
            return _row_to_summary(rows[1], conn)

        return None
    finally:
        conn.close()


def get_run_by_id(
    run_id: str, db_path: str = _DEFAULT_DB_PATH
) -> Optional[EvalRunSummary]:
    """Get a specific run by its ID."""
    init_db(db_path)
    conn = _get_connection(db_path)

    try:
        row = conn.execute(
            "SELECT * FROM eval_runs WHERE run_id = ?", (run_id,)
        ).fetchone()

        if row is None:
            return None

        return _row_to_summary(row, conn)
    finally:
        conn.close()


def get_run_history(
    n: int = 10, db_path: str = _DEFAULT_DB_PATH
) -> list[EvalRunSummary]:
    """Get the last N evaluation runs, ordered oldest to newest."""
    init_db(db_path)
    conn = _get_connection(db_path)

    try:
        rows = conn.execute(
            "SELECT * FROM eval_runs ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()

        # Reverse to get oldest-first order (needed for drift detection)
        runs = []
        for row in reversed(rows):
            runs.append(_row_to_summary(row, conn))
        return runs
    finally:
        conn.close()


def set_baseline(run_id: str, db_path: str = _DEFAULT_DB_PATH) -> None:
    """Mark a specific run as the baseline (unmarks any previous baseline)."""
    init_db(db_path)
    conn = _get_connection(db_path)

    try:
        conn.execute("UPDATE eval_runs SET is_baseline = 0 WHERE is_baseline = 1")
        conn.execute("UPDATE eval_runs SET is_baseline = 1 WHERE run_id = ?", (run_id,))
        conn.commit()
    finally:
        conn.close()


def _row_to_summary(row: sqlite3.Row, conn: sqlite3.Connection) -> EvalRunSummary:
    """Convert a database row to an EvalRunSummary, including results."""
    run_id = row["run_id"]

    # Load per-category metrics
    per_category_data = json.loads(row["per_category_json"] or "[]")
    per_category = [
        CategoryMetrics(
            category=EmailCategory(cm["category"]),
            total=cm["total"],
            correct=cm["correct"],
            accuracy=cm["accuracy"],
            avg_summary_score=cm.get("avg_summary_score", 0.0),
            avg_latency_ms=cm.get("avg_latency_ms", 0.0),
        )
        for cm in per_category_data
    ]

    # Load individual results
    result_rows = conn.execute(
        "SELECT * FROM eval_results WHERE run_id = ? ORDER BY test_case_id",
        (run_id,),
    ).fetchall()

    results = [
        TestCaseResult(
            test_case_id=r["test_case_id"],
            input_email=r["input_email"],
            expected_category=EmailCategory(r["expected_category"]),
            expected_summary=r["expected_summary"],
            predicted_category=EmailCategory(r["predicted_category"]) if r["predicted_category"] else None,
            predicted_summary=r["predicted_summary"] or "",
            category_match=bool(r["category_match"]),
            summary_relevance_score=r["summary_relevance_score"],
            latency_ms=r["latency_ms"],
            tokens_used=r["tokens_used"],
            error=r["error"],
            difficulty=Difficulty(r["difficulty"]) if r["difficulty"] else Difficulty.MEDIUM,
        )
        for r in result_rows
    ]

    # Parse timestamp
    ts = datetime.fromisoformat(row["timestamp"])
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    return EvalRunSummary(
        run_id=run_id,
        prompt_version=row["prompt_version"],
        model=row["model"],
        timestamp=ts,
        total_cases=row["total_cases"],
        passed_cases=row["passed_cases"],
        failed_cases=row["failed_cases"],
        error_cases=row["error_cases"],
        overall_accuracy=row["overall_accuracy"],
        avg_summary_score=row["avg_summary_score"],
        avg_latency_ms=row["avg_latency_ms"],
        total_tokens=row["total_tokens"],
        per_category=per_category,
        results=results,
    )
