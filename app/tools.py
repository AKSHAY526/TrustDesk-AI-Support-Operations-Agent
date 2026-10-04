"""Catalogue-driven tool engine."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .repository import get_action, get_tool

STATUS_APPROVAL_REQUIRED = "approval_required"
STATUS_APPROVED = "approved"
STATUS_EXECUTED = "executed"
STATUS_REJECTED = "rejected"
STATUS_FAILED = "failed"

DECISION_APPROVED = "approved"
DECISION_REJECTED = "rejected"

# Money-like fields the catalogue cap applies to.
_AMOUNT_FIELDS = ("amount", "amount_inr", "value")


class ToolError(Exception):
    """A tool request or approval was rejected. Carries a stable error code."""

    def __init__(self, code: str, message: str, http_status: int = 400, **extra: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.extra = extra

    def to_detail(self) -> dict:
        return {"error": self.code, "message": self.message, **self.extra}


@dataclass
class ActionOutcome:
    """Result of requesting or approving an action."""

    action: dict
    created: bool = False
    executed: bool = False
    replayed: bool = False

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "created": self.created,
            "executed": self.executed,
            "replayed": self.replayed,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _amount_of(payload: dict) -> float | None:
    for field in _AMOUNT_FIELDS:
        if field in payload:
            try:
                return float(payload[field])
            except (TypeError, ValueError):
                raise ToolError(
                    "invalid_payload", f"Field {field!r} must be a number.", 422
                ) from None
    return None


def validate_request(
    conn: sqlite3.Connection,
    tool_name: str,
    payload: dict,
    ticket_category: str | None,
) -> dict:
    """Check a proposed action against the catalogue. Returns the tool row."""
    tool = get_tool(conn, tool_name)
    if tool is None:
        raise ToolError(
            "unknown_tool",
            f"{tool_name!r} is not in the tool catalogue.",
            404,
            tool_name=tool_name,
        )

    missing = [field for field in tool["required_fields"] if not payload.get(field)]
    if missing:
        raise ToolError(
            "missing_required_fields",
            f"{tool_name} requires {missing}.",
            422,
            missing=missing,
        )

    allowed = tool["allowed_categories"]
    if allowed and ticket_category not in allowed:
        raise ToolError(
            "tool_not_allowed_for_category",
            f"{tool_name} may only be used for {allowed}, not {ticket_category!r}.",
            403,
            allowed_categories=allowed,
            ticket_category=ticket_category,
        )

    cap = tool.get("max_amount_inr")
    amount = _amount_of(payload)
    if cap is not None and amount is not None and amount > float(cap):
        raise ToolError(
            "amount_over_limit",
            f"{tool_name} is capped at {cap} INR; {amount} was requested.",
            403,
            cap=float(cap),
            requested=amount,
        )

    return tool


def request_action(
    conn: sqlite3.Connection,
    *,
    ticket_id: str,
    tool_name: str,
    payload: dict,
    ticket_category: str | None,
    requested_by: str,
) -> ActionOutcome:
    """Register an action request."""
    tool = validate_request(conn, tool_name, payload, ticket_category)
    idempotency_key = str(payload["idempotency_key"])

    existing = conn.execute(
        "SELECT action_id FROM tool_actions WHERE idempotency_key = ?",
        (idempotency_key,),
    ).fetchone()
    if existing is not None:
        return ActionOutcome(action=get_action(conn, existing["action_id"]) or {}, replayed=True)

    needs_approval = bool(tool["requires_human_approval"])
    action_id = _new_id("act")

    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """INSERT INTO tool_actions
               (action_id, ticket_id, tool_name, payload, risk_level,
                requires_human_approval, status, idempotency_key, requested_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                action_id, ticket_id, tool_name, json.dumps(payload), tool.get("risk_level"),
                int(needs_approval),
                STATUS_APPROVAL_REQUIRED if needs_approval else STATUS_APPROVED,
                idempotency_key, requested_by, _now(),
            ),
        )
        conn.execute("COMMIT")
    except sqlite3.IntegrityError:
        # Another request won the race on the same idempotency key.
        conn.execute("ROLLBACK")
        row = conn.execute(
            "SELECT action_id FROM tool_actions WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if row is None:
            raise
        return ActionOutcome(action=get_action(conn, row["action_id"]) or {}, replayed=True)
    except Exception:
        conn.execute("ROLLBACK")
        raise

    if needs_approval:
        return ActionOutcome(action=get_action(conn, action_id) or {}, created=True)

    executed = _claim_and_execute(conn, action_id)
    return ActionOutcome(action=get_action(conn, action_id) or {}, created=True, executed=executed)


def decide_action(
    conn: sqlite3.Connection,
    *,
    action_id: str,
    reviewer_id: str,
    decision: str,
    reason: str | None = None,
) -> ActionOutcome:
    """Record a manager's approval or rejection, then execute if approved."""
    if decision not in (DECISION_APPROVED, DECISION_REJECTED):
        raise ToolError("invalid_decision", "Decision must be 'approved' or 'rejected'.", 422)

    action = get_action(conn, action_id)
    if action is None:
        raise ToolError("action_not_found", f"No action {action_id!r}.", 404)

    if action["status"] == STATUS_EXECUTED:
        # Approving twice is a no-op, not a second side effect.
        return ActionOutcome(action=action, replayed=True)
    if action["status"] == STATUS_REJECTED:
        raise ToolError("action_already_rejected", "This action was already rejected.", 409)
    if action["status"] not in (STATUS_APPROVAL_REQUIRED, STATUS_APPROVED):
        raise ToolError(
            "action_not_pending",
            f"Action is in state {action['status']!r} and cannot be decided.",
            409,
        )

    if action.get("requested_by") and action["requested_by"] == reviewer_id:
        raise ToolError(
            "separation_of_duties",
            "The person who requested an action may not approve it.",
            403,
            requested_by=action["requested_by"],
        )

    conn.execute(
        """INSERT INTO approvals (approval_id, action_id, reviewer_id, decision, reason, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (_new_id("apr"), action_id, reviewer_id, decision, reason, _now()),
    )

    if decision == DECISION_REJECTED:
        conn.execute(
            "UPDATE tool_actions SET status = ? WHERE action_id = ? AND status IN (?, ?)",
            (STATUS_REJECTED, action_id, STATUS_APPROVAL_REQUIRED, STATUS_APPROVED),
        )
        return ActionOutcome(action=get_action(conn, action_id) or {})

    executed = _claim_and_execute(conn, action_id)
    return ActionOutcome(action=get_action(conn, action_id) or {}, executed=executed)


def _claim_and_execute(conn: sqlite3.Connection, action_id: str) -> bool:
    """Move an action to ``executed`` exactly once, then perform the effect."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        # The conditional UPDATE is the claim: only the caller that changes the
        # row goes on to perform the side effect.
        cursor = conn.execute(
            """UPDATE tool_actions SET status = ?, executed_at = ?
               WHERE action_id = ? AND status IN (?, ?)""",
            (STATUS_EXECUTED, _now(), action_id, STATUS_APPROVAL_REQUIRED, STATUS_APPROVED),
        )
        claimed = cursor.rowcount == 1
        if claimed:
            row = conn.execute(
                "SELECT tool_name, payload, ticket_id FROM tool_actions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
            result = _perform_side_effect(
                row["tool_name"], json.loads(row["payload"] or "{}"), row["ticket_id"]
            )
            conn.execute(
                "UPDATE tool_actions SET result = ? WHERE action_id = ?",
                (json.dumps(result), action_id),
            )
        conn.execute("COMMIT")
        return claimed
    except Exception:
        conn.execute("ROLLBACK")
        conn.execute(
            "UPDATE tool_actions SET status = ? WHERE action_id = ? AND status != ?",
            (STATUS_FAILED, action_id, STATUS_EXECUTED),
        )
        raise


def _perform_side_effect(tool_name: str, payload: dict, ticket_id: str) -> dict:
    """Mock side effects. Nothing external is called."""
    reference = f"{tool_name}:{payload.get('idempotency_key')}"
    base = {"tool": tool_name, "ticket_id": ticket_id, "reference": reference, "at": _now()}

    if tool_name == "create_replacement_order":
        return {**base, "replacement_order_id": _new_id("ord_repl"), "sku": payload.get("sku")}
    if tool_name == "start_refund_review":
        return {**base, "refund_review_id": _new_id("rev"), "amount": payload.get("amount")}
    if tool_name == "issue_coupon":
        return {**base, "coupon_code": _new_id("cpn").upper(), "amount": payload.get("amount")}
    if tool_name == "open_carrier_investigation":
        return {**base, "investigation_id": _new_id("inv"),
                "tracking_number": payload.get("tracking_number")}
    if tool_name == "escalate_to_human":
        return {**base, "escalation_id": _new_id("esc"), "queue": payload.get("queue")}
    if tool_name == "lock_account":
        return {**base, "lock_id": _new_id("lck"), "customer_id": payload.get("customer_id")}
    return {**base, "note": "no-op tool"}
