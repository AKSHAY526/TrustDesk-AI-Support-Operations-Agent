"""The tool engine: catalogue rules, approval, separation of duties, exactly-once."""

from __future__ import annotations

import threading

import pytest

from app import tools
from app.db import connect
from app.repository import get_action

AGENT_ID = "agent@demo"
MANAGER_ID = "manager@demo"


def _replacement_payload(key: str = "tkt_9001-replacement-1") -> dict:
    return {
        "order_id": "ord_5001",
        "sku": "BG-AIRPODS-01",
        "reason": "Damaged on arrival",
        "idempotency_key": key,
    }


# --- catalogue-driven validation -------------------------------------------

def test_catalogue_supplies_the_rules_for_all_six_tools(conn):
    from app.repository import list_tools

    catalogue = {tool["tool_name"]: tool for tool in list_tools(conn)}
    assert set(catalogue) == {
        "create_replacement_order", "start_refund_review", "issue_coupon",
        "open_carrier_investigation", "escalate_to_human", "lock_account",
    }
    assert catalogue["issue_coupon"]["max_amount_inr"] == 1000
    assert catalogue["escalate_to_human"]["requires_human_approval"] is False
    assert catalogue["lock_account"]["requires_human_approval"] is True


def test_unknown_tool_is_rejected(conn):
    with pytest.raises(tools.ToolError) as exc:
        tools.request_action(
            conn, ticket_id="tkt_9001", tool_name="wire_money",
            payload={"idempotency_key": "k1"}, ticket_category="refund",
            requested_by=AGENT_ID,
        )
    assert exc.value.code == "unknown_tool"


def test_missing_required_field_is_rejected(conn):
    with pytest.raises(tools.ToolError) as exc:
        tools.request_action(
            conn, ticket_id="tkt_9001", tool_name="create_replacement_order",
            payload={"order_id": "ord_5001", "idempotency_key": "k2"},
            ticket_category="refund", requested_by=AGENT_ID,
        )
    assert exc.value.code == "missing_required_fields"
    assert "sku" in exc.value.extra["missing"]


def test_tool_is_refused_for_a_category_the_catalogue_does_not_allow(conn):
    """create_replacement_order is refund/warranty only."""
    with pytest.raises(tools.ToolError) as exc:
        tools.request_action(
            conn, ticket_id="tkt_9008", tool_name="create_replacement_order",
            payload=_replacement_payload("k3"), ticket_category="billing",
            requested_by=AGENT_ID,
        )
    assert exc.value.code == "tool_not_allowed_for_category"


def test_amount_above_the_catalogue_cap_is_refused(conn):
    """The coupon cap is 1000 INR, so the classic 5000 request cannot pass."""
    with pytest.raises(tools.ToolError) as exc:
        tools.request_action(
            conn, ticket_id="tkt_9006", tool_name="issue_coupon",
            payload={"customer_id": "cus_1006", "amount": 5000,
                     "reason": "goodwill", "idempotency_key": "k4"},
            ticket_category="general", requested_by=AGENT_ID,
        )
    assert exc.value.code == "amount_over_limit"
    assert exc.value.extra["cap"] == 1000.0


def test_amount_within_the_cap_is_allowed(conn):
    outcome = tools.request_action(
        conn, ticket_id="tkt_9006", tool_name="issue_coupon",
        payload={"customer_id": "cus_1006", "amount": 500,
                 "reason": "goodwill", "idempotency_key": "k5"},
        ticket_category="general", requested_by=AGENT_ID,
    )
    assert outcome.action["status"] == tools.STATUS_APPROVAL_REQUIRED


# --- approval -------------------------------------------------------------

def test_low_risk_tool_executes_without_approval(conn):
    outcome = tools.request_action(
        conn, ticket_id="tkt_9002", tool_name="open_carrier_investigation",
        payload={"order_id": "ord_5002", "tracking_number": "BLUETRK10002",
                 "reason": "no movement", "idempotency_key": "k6"},
        ticket_category="shipping", requested_by=AGENT_ID,
    )
    assert outcome.executed
    assert outcome.action["status"] == tools.STATUS_EXECUTED
    assert outcome.action["result"]["investigation_id"]


def test_risky_tool_waits_and_does_not_act(conn):
    outcome = tools.request_action(
        conn, ticket_id="tkt_9001", tool_name="create_replacement_order",
        payload=_replacement_payload(), ticket_category="refund", requested_by=AGENT_ID,
    )
    assert not outcome.executed
    assert outcome.action["status"] == tools.STATUS_APPROVAL_REQUIRED
    assert outcome.action["result"] is None, "nothing may happen before approval"


def test_approval_executes_and_records_the_reviewer(conn):
    requested = tools.request_action(
        conn, ticket_id="tkt_9001", tool_name="create_replacement_order",
        payload=_replacement_payload(), ticket_category="refund", requested_by=AGENT_ID,
    )
    decided = tools.decide_action(
        conn, action_id=requested.action["action_id"], reviewer_id=MANAGER_ID,
        decision=tools.DECISION_APPROVED, reason="photo verified",
    )
    assert decided.executed
    assert decided.action["status"] == tools.STATUS_EXECUTED
    assert decided.action["result"]["replacement_order_id"]
    assert decided.action["approvals"][0]["reviewer_id"] == MANAGER_ID


def test_rejection_prevents_execution(conn):
    requested = tools.request_action(
        conn, ticket_id="tkt_9001", tool_name="create_replacement_order",
        payload=_replacement_payload(), ticket_category="refund", requested_by=AGENT_ID,
    )
    decided = tools.decide_action(
        conn, action_id=requested.action["action_id"], reviewer_id=MANAGER_ID,
        decision=tools.DECISION_REJECTED, reason="no evidence of damage",
    )
    assert not decided.executed
    assert decided.action["status"] == tools.STATUS_REJECTED
    assert decided.action["result"] is None


def test_requester_cannot_approve_their_own_action(conn):
    """Separation of duties."""
    requested = tools.request_action(
        conn, ticket_id="tkt_9001", tool_name="create_replacement_order",
        payload=_replacement_payload(), ticket_category="refund", requested_by=AGENT_ID,
    )
    with pytest.raises(tools.ToolError) as exc:
        tools.decide_action(
            conn, action_id=requested.action["action_id"], reviewer_id=AGENT_ID,
            decision=tools.DECISION_APPROVED,
        )
    assert exc.value.code == "separation_of_duties"
    assert get_action(conn, requested.action["action_id"])["status"] == tools.STATUS_APPROVAL_REQUIRED


# --- exactly once ---------------------------------------------------------

def test_same_idempotency_key_does_not_create_a_second_action(conn):
    first = tools.request_action(
        conn, ticket_id="tkt_9001", tool_name="create_replacement_order",
        payload=_replacement_payload(), ticket_category="refund", requested_by=AGENT_ID,
    )
    second = tools.request_action(
        conn, ticket_id="tkt_9001", tool_name="create_replacement_order",
        payload=_replacement_payload(), ticket_category="refund", requested_by=AGENT_ID,
    )
    assert second.replayed
    assert second.action["action_id"] == first.action["action_id"]
    assert conn.execute("SELECT COUNT(*) FROM tool_actions").fetchone()[0] == 1


def test_approving_twice_does_not_run_the_effect_twice(conn):
    requested = tools.request_action(
        conn, ticket_id="tkt_9001", tool_name="create_replacement_order",
        payload=_replacement_payload(), ticket_category="refund", requested_by=AGENT_ID,
    )
    action_id = requested.action["action_id"]
    first = tools.decide_action(
        conn, action_id=action_id, reviewer_id=MANAGER_ID, decision=tools.DECISION_APPROVED
    )
    second = tools.decide_action(
        conn, action_id=action_id, reviewer_id=MANAGER_ID, decision=tools.DECISION_APPROVED
    )
    assert first.executed
    assert second.replayed and not second.executed
    assert first.action["result"] == get_action(conn, action_id)["result"]


def test_concurrent_approvals_produce_exactly_one_side_effect(seeded_db):
    """Ten managers click approve at the same instant. One replacement only."""
    setup = connect(seeded_db)
    requested = tools.request_action(
        setup, ticket_id="tkt_9001", tool_name="create_replacement_order",
        payload=_replacement_payload(), ticket_category="refund", requested_by=AGENT_ID,
    )
    action_id = requested.action["action_id"]
    setup.close()

    executed_flags: list[bool] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(10)

    def approve(index: int) -> None:
        connection = connect(seeded_db)
        try:
            barrier.wait(timeout=10)
            outcome = tools.decide_action(
                connection, action_id=action_id, reviewer_id=f"manager{index}@demo",
                decision=tools.DECISION_APPROVED,
            )
            executed_flags.append(outcome.executed)
        except Exception as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)
        finally:
            connection.close()

    threads = [threading.Thread(target=approve, args=(i,)) for i in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, f"unexpected errors: {errors}"
    assert sum(executed_flags) == 1, "exactly one caller may perform the side effect"

    check = connect(seeded_db)
    try:
        action = get_action(check, action_id)
        assert action["status"] == tools.STATUS_EXECUTED
        assert check.execute("SELECT COUNT(*) FROM tool_actions").fetchone()[0] == 1
        # One replacement order id, not ten.
        assert action["result"]["replacement_order_id"]
    finally:
        check.close()


def test_idempotency_key_is_unique_at_the_database_level(conn):
    """The guarantee lives in the schema, not only in application code."""
    import sqlite3

    tools.request_action(
        conn, ticket_id="tkt_9001", tool_name="create_replacement_order",
        payload=_replacement_payload("shared-key"), ticket_category="refund",
        requested_by=AGENT_ID,
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO tool_actions
               (action_id, ticket_id, tool_name, status, idempotency_key)
               VALUES ('act_manual', 'tkt_9001', 'create_replacement_order', 'approved', 'shared-key')"""
        )
