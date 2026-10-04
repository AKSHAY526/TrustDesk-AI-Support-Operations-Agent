"""Run traces: the audit record for every AI decision."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from .config import PROMPT_VERSION

RUN_TRIAGE = "triage"
RUN_DRAFT = "draft_reply"
RUN_TOOL = "tool_recommendation"
RUN_EVAL = "eval_case"


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    ticket_id: str | None,
    run_type: str,
    status: str,
    retrieved_doc_ids: list[str] | None = None,
    tool_calls: list[dict] | None = None,
    guardrail_results: dict | None = None,
    escalation_reason: str | None = None,
    model_name: str | None = None,
    latency_ms: int | None = None,
) -> str:
    """Persist one trace row and return its run id."""
    conn.execute(
        """INSERT OR REPLACE INTO traces
           (run_id, ticket_id, run_type, status, retrieved_doc_ids, tool_calls,
            guardrail_results, escalation_reason, model_name, prompt_version,
            latency_ms, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            ticket_id,
            run_type,
            status,
            json.dumps(retrieved_doc_ids or []),
            json.dumps(tool_calls or []),
            json.dumps(guardrail_results or {}),
            escalation_reason,
            model_name,
            PROMPT_VERSION,
            latency_ms,
            _now(),
        ),
    )
    return run_id


def get(conn: sqlite3.Connection, run_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM traces WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    trace = dict(row)
    trace["retrieved_doc_ids"] = json.loads(trace.get("retrieved_doc_ids") or "[]")
    trace["tool_calls"] = json.loads(trace.get("tool_calls") or "[]")
    trace["guardrail_results"] = json.loads(trace.get("guardrail_results") or "{}")
    return trace


def list_for_ticket(conn: sqlite3.Connection, ticket_id: str, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        """SELECT run_id FROM traces WHERE ticket_id = ?
           ORDER BY created_at DESC, rowid DESC LIMIT ?""",
        (ticket_id, limit),
    ).fetchall()
    return [t for t in (get(conn, row["run_id"]) for row in rows) if t]


def escalation_gaps(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    """Escalations with their reasons - the policy-gap report."""
    rows = conn.execute(
        """SELECT ticket_id, run_type, escalation_reason, created_at
           FROM traces
           WHERE escalation_reason IS NOT NULL AND escalation_reason != ''
           ORDER BY created_at DESC, rowid DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]
