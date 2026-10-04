"""Tool action endpoints: request, list, and manager approval."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query, status

from .. import repository, tools
from ..auth import Principal, current_principal, require_approver, require_manager
from ..deps import get_db
from ..models import ActionDecision, ActionRequest, ActionResponse

router = APIRouter(prefix="/actions", tags=["actions"])


@router.get("/catalogue", summary="List the tool catalogue and its rules")
def catalogue(
    conn: sqlite3.Connection = Depends(get_db),
    principal: Principal = Depends(current_principal),
) -> list[dict]:
    return repository.list_tools(conn)


@router.get("", summary="List requested actions")
def list_actions(
    ticket_id: str | None = None,
    action_status: str | None = Query(default=None, alias="status"),
    conn: sqlite3.Connection = Depends(get_db),
    principal: Principal = Depends(current_principal),
) -> list[dict]:
    """Anyone can list one ticket's actions. Listing across every ticket is the
    approval queue (for example ?status=approval_required), so only managers and admins can.
    """
    if not ticket_id:
        require_manager(principal)
    return repository.list_actions(conn, ticket_id, action_status)


@router.post("", response_model=ActionResponse, summary="Request a tool action")
def request_action(
    payload: ActionRequest,
    conn: sqlite3.Connection = Depends(get_db),
    principal: Principal = Depends(current_principal),
) -> dict:
    """Validate against the catalogue, then either execute or hold for approval."""
    ticket = repository.get_ticket(conn, payload.ticket_id)
    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "ticket_not_found", "message": f"No ticket {payload.ticket_id!r}."},
        )

    triage = repository.latest_triage(conn, payload.ticket_id)
    category = triage["category"] if triage else None

    try:
        outcome = tools.request_action(
            conn,
            ticket_id=payload.ticket_id,
            tool_name=payload.tool_name,
            payload=payload.payload,
            ticket_category=category,
            requested_by=principal.user_id,
        )
    except tools.ToolError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.to_detail()) from exc
    return outcome.to_dict()


@router.get("/{action_id}", summary="Fetch one action")
def get_action(
    action_id: str,
    conn: sqlite3.Connection = Depends(get_db),
    principal: Principal = Depends(current_principal),
) -> dict:
    action = repository.get_action(conn, action_id)
    if action is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "action_not_found", "message": f"No action {action_id!r}."},
        )
    return action


@router.post("/{action_id}/approve", response_model=ActionResponse, summary="Approve or reject (manager)")
def decide_action(
    action_id: str,
    decision: ActionDecision,
    conn: sqlite3.Connection = Depends(get_db),
    principal: Principal = Depends(require_approver),
) -> dict:
    """Manager-only. The requester may not approve their own action."""
    try:
        outcome = tools.decide_action(
            conn,
            action_id=action_id,
            reviewer_id=principal.user_id,
            decision=decision.decision,
            reason=decision.reason,
        )
    except tools.ToolError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.to_detail()) from exc
    return outcome.to_dict()
