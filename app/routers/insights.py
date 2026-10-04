"""Trace and evaluation endpoints."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status

from .. import evals, traces
from ..auth import Principal, current_principal, require_admin, require_manager
from ..db import init_db
from ..deps import get_db
from ..models import EvalQueuedResponse, EvalRunResponse

router = APIRouter(tags=["traces and evaluation"])


@router.get("/traces/{run_id}", summary="Fetch one run trace")
def get_trace(
    run_id: str,
    conn: sqlite3.Connection = Depends(get_db),
    principal: Principal = Depends(current_principal),
) -> dict:
    trace = traces.get(conn, run_id)
    if trace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "trace_not_found", "message": f"No run {run_id!r}."},
        )
    return trace


@router.get("/tickets/{ticket_id}/traces", summary="Traces for a ticket")
def ticket_traces(
    ticket_id: str,
    conn: sqlite3.Connection = Depends(get_db),
    principal: Principal = Depends(current_principal),
) -> list[dict]:
    return traces.list_for_ticket(conn, ticket_id)


@router.get("/insights/policy-gaps", summary="Escalation reasons, i.e. missing policies (manager)")
def policy_gaps(
    limit: int = Query(default=50, ge=1, le=500),
    conn: sqlite3.Connection = Depends(get_db),
    principal: Principal = Depends(require_manager),
) -> list[dict]:
    """Questions the agent could not answer, newest first. Managers and admins only."""
    return traces.escalation_gaps(conn, limit)


def _run_evals_background() -> None:
    conn = init_db()
    try:
        evals.run(conn)
    finally:
        conn.close()


@router.post(
    "/evals/run",
    response_model=EvalRunResponse | EvalQueuedResponse,
    summary="Run the evaluation set (admin)",
)
def run_evals(
    background: BackgroundTasks,
    run_in_background: bool = Query(
        default=False,
        alias="background",
        description="Return immediately and run the suite in the background.",
    ),
    conn: sqlite3.Connection = Depends(get_db),
    principal: Principal = Depends(require_admin),
) -> dict:
    if run_in_background:
        background.add_task(_run_evals_background)
        return {"status": "queued", "message": "Evaluation started; poll /evals/latest for the report."}
    return evals.run(conn)


@router.get("/evals/latest", response_model=EvalRunResponse, summary="Latest evaluation report (admin)")
def latest_eval(
    conn: sqlite3.Connection = Depends(get_db),
    principal: Principal = Depends(require_admin),
) -> dict:
    report = evals.latest(conn)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "no_eval_runs", "message": "No evaluation has been run yet."},
        )
    return report
