"""Read queries for tickets, customers, orders and tools."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

# Seed-only evaluation labels. Production logic must never read these.
_EXPECTED_FIELDS = (
    "expected_category",
    "expected_priority",
    "expected_sentiment",
    "expected_escalation",
    "expected_actions",
)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _decode(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def list_tickets(conn: sqlite3.Connection, status: str | None = None) -> list[dict]:
    sql = """SELECT ticket_id, customer_id, order_id, channel, subject, created_at, status
             FROM tickets"""
    params: tuple = ()
    if status:
        sql += " WHERE status = ?"
        params = (status,)
    sql += " ORDER BY ticket_id"
    return [dict(row) for row in conn.execute(sql, params)]


def get_ticket(conn: sqlite3.Connection, ticket_id: str, include_expected: bool = False) -> dict | None:
    row = conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
    ticket = _row_to_dict(row)
    if ticket is None:
        return None
    ticket["expected_actions"] = _decode(ticket.get("expected_actions"), [])
    if ticket.get("expected_escalation") is not None:
        ticket["expected_escalation"] = bool(ticket["expected_escalation"])
    if not include_expected:
        for field in _EXPECTED_FIELDS:
            ticket.pop(field, None)
    return ticket


def get_customer(conn: sqlite3.Connection, customer_id: str | None) -> dict | None:
    if not customer_id:
        return None
    customer = _row_to_dict(
        conn.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
    )
    if customer is not None:
        customer["tags"] = _decode(customer.get("tags"), [])
        customer["verified"] = bool(customer.get("verified"))
    return customer


def get_order(conn: sqlite3.Connection, order_id: str | None) -> dict | None:
    if not order_id:
        return None
    order = _row_to_dict(
        conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    )
    if order is not None:
        order["items"] = _decode(order.get("items"), [])
    return order


def get_ticket_context(conn: sqlite3.Connection, ticket_id: str) -> dict | None:
    """Ticket plus only its own customer and order."""
    ticket = get_ticket(conn, ticket_id)
    if ticket is None:
        return None
    return {
        "ticket": ticket,
        "customer": get_customer(conn, ticket.get("customer_id")),
        "order": get_order(conn, ticket.get("order_id")),
    }


def get_tool(conn: sqlite3.Connection, tool_name: str) -> dict | None:
    row = conn.execute("SELECT * FROM tool_catalog WHERE tool_name = ?", (tool_name,)).fetchone()
    tool = _row_to_dict(row)
    if tool is None:
        return None
    tool["allowed_categories"] = _decode(tool.get("allowed_categories"), [])
    tool["required_fields"] = _decode(tool.get("required_fields"), [])
    tool["requires_human_approval"] = bool(tool.get("requires_human_approval"))
    return tool


def list_tools(conn: sqlite3.Connection) -> list[dict]:
    names = [row["tool_name"] for row in conn.execute("SELECT tool_name FROM tool_catalog ORDER BY tool_name")]
    return [tool for tool in (get_tool(conn, name) for name in names) if tool]


def latest_triage(conn: sqlite3.Connection, ticket_id: str) -> dict | None:
    row = conn.execute(
        """SELECT * FROM triage_results WHERE ticket_id = ?
           ORDER BY created_at DESC, rowid DESC LIMIT 1""",
        (ticket_id,),
    ).fetchone()
    triage = _row_to_dict(row)
    if triage is not None:
        triage["should_escalate"] = bool(triage.get("should_escalate"))
    return triage


def list_drafts(conn: sqlite3.Connection, ticket_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM drafts WHERE ticket_id = ? ORDER BY created_at DESC, rowid DESC",
        (ticket_id,),
    ).fetchall()
    drafts = []
    for row in rows:
        draft = dict(row)
        draft["citations"] = _decode(draft.get("citations"), [])
        drafts.append(draft)
    return drafts


def get_action(conn: sqlite3.Connection, action_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM tool_actions WHERE action_id = ?", (action_id,)).fetchone()
    action = _row_to_dict(row)
    if action is None:
        return None
    action["payload"] = _decode(action.get("payload"), {})
    action["result"] = _decode(action.get("result"), None)
    action["requires_human_approval"] = bool(action.get("requires_human_approval"))
    action["approvals"] = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM approvals WHERE action_id = ? ORDER BY created_at", (action_id,)
        )
    ]
    return action


def list_actions(
    conn: sqlite3.Connection, ticket_id: str | None = None, status: str | None = None
) -> list[dict]:
    sql = "SELECT action_id FROM tool_actions"
    clauses: list[str] = []
    params: list[str] = []
    if ticket_id:
        clauses.append("ticket_id = ?")
        params.append(ticket_id)
    if status:
        clauses.append("status = ?")
        params.append(status)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC, rowid DESC"
    ids = [row["action_id"] for row in conn.execute(sql, params)]
    return [action for action in (get_action(conn, aid) for aid in ids) if action]
