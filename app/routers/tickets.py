"""Ticket endpoints: queue, detail, triage and cited drafts."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query, status

from .. import agent, repository
from ..auth import Principal, current_principal
from ..deps import get_db
from ..models import DraftResponse, TicketContext, TicketSummary, TriageResponse

router = APIRouter(prefix="/tickets", tags=["tickets"])


def _ticket_or_404(conn: sqlite3.Connection, ticket_id: str) -> dict:
    ticket = repository.get_ticket(conn, ticket_id)
    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "ticket_not_found", "message": f"No ticket {ticket_id!r}."},
        )
    return ticket


@router.get("", response_model=list[TicketSummary], summary="List tickets")
def list_tickets(
    ticket_status: str | None = Query(default=None, alias="status"),
    conn: sqlite3.Connection = Depends(get_db),
    principal: Principal = Depends(current_principal),
) -> list[dict]:
    return repository.list_tickets(conn, ticket_status)


@router.get("/{ticket_id}", response_model=TicketContext, summary="Open a ticket with its context")
def get_ticket(
    ticket_id: str,
    conn: sqlite3.Connection = Depends(get_db),
    principal: Principal = Depends(current_principal),
) -> dict:
    _ticket_or_404(conn, ticket_id)
    # Only this ticket's customer and order, never anyone else's.
    context = repository.get_ticket_context(conn, ticket_id) or {}
    return {
        **context,
        "triage": repository.latest_triage(conn, ticket_id),
        "drafts": repository.list_drafts(conn, ticket_id),
        "actions": repository.list_actions(conn, ticket_id),
    }


@router.post("/{ticket_id}/triage", response_model=TriageResponse, summary="Triage a ticket")
def triage_ticket(
    ticket_id: str,
    conn: sqlite3.Connection = Depends(get_db),
    principal: Principal = Depends(current_principal),
) -> dict:
    _ticket_or_404(conn, ticket_id)
    outcome = agent.triage(conn, ticket_id)
    if outcome is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "ticket_not_found", "message": f"No ticket {ticket_id!r}."},
        )
    return outcome.to_dict()


@router.post("/{ticket_id}/draft", response_model=DraftResponse, summary="Draft a cited reply")
def draft_ticket_reply(
    ticket_id: str,
    conn: sqlite3.Connection = Depends(get_db),
    principal: Principal = Depends(current_principal),
) -> dict:
    _ticket_or_404(conn, ticket_id)
    if repository.latest_triage(conn, ticket_id) is None:
        # Drafting depends on the category, so triage first rather than guessing.
        agent.triage(conn, ticket_id)
    outcome = agent.draft_reply(conn, ticket_id)
    if outcome is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "ticket_not_found", "message": f"No ticket {ticket_id!r}."},
        )
    return outcome.to_dict()
