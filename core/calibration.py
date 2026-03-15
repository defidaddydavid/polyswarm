"""
Calibration tracker using Brier scores.
Brier score = mean((forecast - outcome)^2), lower is better.
Perfect calibration = 0.0, random = 0.25.
"""

from __future__ import annotations
import sqlite3
import os
from datetime import datetime


DB_PATH = os.getenv("DATABASE_URL", "polyswarm.db").replace("sqlite+aiosqlite:///./", "")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            agent_id TEXT,
            probability REAL NOT NULL,
            outcome REAL,
            brier_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            resolved_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS swarm_forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            final_probability REAL NOT NULL,
            consensus_score REAL,
            outcome REAL,
            brier_score REAL,
            market_odds REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            resolved_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_forecast(question: str, agent_id: str, probability: float, market_odds: float | None = None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO forecasts (question, agent_id, probability) VALUES (?, ?, ?)",
        (question, agent_id, probability),
    )
    conn.commit()
    conn.close()


def save_swarm_forecast(question: str, probability: float, consensus_score: float, market_odds: float | None = None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO swarm_forecasts (question, final_probability, consensus_score, market_odds) VALUES (?, ?, ?, ?)",
        (question, probability, consensus_score, market_odds),
    )
    conn.commit()
    conn.close()


def resolve_forecast(question: str, outcome: float):
    """outcome: 1.0 = YES resolved, 0.0 = NO resolved"""
    conn = sqlite3.connect(DB_PATH)
    # update individual agent forecasts
    rows = conn.execute(
        "SELECT id, probability FROM forecasts WHERE question = ? AND outcome IS NULL",
        (question,)
    ).fetchall()
    for row_id, prob in rows:
        brier = (prob - outcome) ** 2
        conn.execute(
            "UPDATE forecasts SET outcome=?, brier_score=?, resolved_at=? WHERE id=?",
            (outcome, brier, datetime.utcnow().isoformat(), row_id),
        )
    # update swarm forecast
    swarm_rows = conn.execute(
        "SELECT id, final_probability FROM swarm_forecasts WHERE question = ? AND outcome IS NULL",
        (question,)
    ).fetchall()
    for row_id, prob in swarm_rows:
        brier = (prob - outcome) ** 2
        conn.execute(
            "UPDATE swarm_forecasts SET outcome=?, brier_score=?, resolved_at=? WHERE id=?",
            (outcome, brier, datetime.utcnow().isoformat(), row_id),
        )
    conn.commit()
    conn.close()


def get_agent_brier_scores() -> dict[str, float]:
    """Returns average Brier score per agent (lower = better calibrated)."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT agent_id, AVG(brier_score) FROM forecasts WHERE brier_score IS NOT NULL GROUP BY agent_id"
    ).fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def get_swarm_brier_score() -> float | None:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT AVG(brier_score) FROM swarm_forecasts WHERE brier_score IS NOT NULL"
    ).fetchone()
    conn.close()
    return row[0] if row else None


def get_calibration_weights() -> dict[str, float]:
    """Convert Brier scores to weights — better calibrated agents get higher weight."""
    scores = get_agent_brier_scores()
    if not scores:
        return {}
    # invert: lower brier = higher weight, normalize
    inverted = {k: 1.0 / (v + 0.01) for k, v in scores.items()}
    max_w = max(inverted.values())
    return {k: v / max_w for k, v in inverted.items()}


def get_forecast_history(limit: int = 50) -> list[dict]:
    """Retrieve past swarm forecasts."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT question, final_probability, consensus_score,
                  market_odds, outcome, brier_score, created_at, resolved_at
           FROM swarm_forecasts ORDER BY created_at DESC LIMIT ?""",
        (limit,)
    ).fetchall()
    conn.close()
    return [
        {
            "question": r[0],
            "probability": r[1],
            "consensus_score": r[2],
            "market_odds": r[3],
            "outcome": r[4],
            "brier_score": r[5],
            "created_at": r[6],
            "resolved_at": r[7],
            "status": "resolved" if r[4] is not None else "pending",
        }
        for r in rows
    ]


def export_calibration(format: str = "json") -> str:
    """Export calibration data as JSON or CSV."""
    agent_scores = get_agent_brier_scores()
    swarm_score = get_swarm_brier_score()
    history = get_forecast_history(limit=1000)

    if format == "csv":
        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["question", "probability", "consensus_score", "market_odds",
                         "outcome", "brier_score", "created_at", "resolved_at", "status"])
        for h in history:
            writer.writerow([h["question"], h["probability"], h["consensus_score"],
                             h["market_odds"], h["outcome"], h["brier_score"],
                             h["created_at"], h["resolved_at"], h["status"]])
        return output.getvalue()
    else:
        import json
        return json.dumps({
            "swarm_brier_score": swarm_score,
            "agent_brier_scores": agent_scores,
            "forecasts": history,
        }, indent=2)
