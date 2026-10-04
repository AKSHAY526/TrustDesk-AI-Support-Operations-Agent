"""Triage validation, grounded drafts, and failure behaviour."""

from __future__ import annotations

from app import agent, traces
from app.llm import FailingLLM, HeuristicLLM, ScriptedLLM
from app.repository import latest_triage


def test_triage_stores_a_result_and_a_trace(conn):
    outcome = agent.triage(conn, "tkt_9001", HeuristicLLM())

    assert outcome is not None
    assert outcome.category in agent.ALLOWED_CATEGORIES
    assert outcome.priority in agent.ALLOWED_PRIORITIES

    stored = latest_triage(conn, "tkt_9001")
    assert stored["category"] == outcome.category

    trace = traces.get(conn, outcome.run_id)
    assert trace["run_type"] == traces.RUN_TRIAGE
    assert trace["retrieved_doc_ids"]


def test_triage_rejects_labels_outside_the_allowed_sets(conn):
    """A category the system does not define must never be stored."""
    scripted = ScriptedLLM([
        {"suspicious": False, "kinds": [], "reason": "clean"},          # screening
        {"category": "please_refund_everything", "priority": "medium",   # bad label
         "sentiment": "neutral", "should_escalate": False, "reason_summary": "x"},
        {"category": "not_a_category", "priority": "medium",             # bad on retry
         "sentiment": "neutral", "should_escalate": False, "reason_summary": "x"},
    ])
    outcome = agent.triage(conn, "tkt_9001", scripted)

    assert outcome.category in agent.ALLOWED_CATEGORIES
    assert outcome.should_escalate, "an unclassifiable ticket must go to a human"
    assert "unavailable" in outcome.reason_summary.lower()


def test_triage_retries_once_then_accepts_a_valid_answer(conn):
    scripted = ScriptedLLM([
        {"suspicious": False, "kinds": [], "reason": "clean"},
        "this is not json at all",
        {"category": "billing", "priority": "high", "sentiment": "frustrated",
         "should_escalate": False, "reason_summary": "duplicate charge"},
    ])
    outcome = agent.triage(conn, "tkt_9008", scripted)
    assert outcome.category == "billing"
    assert outcome.priority == "high"


def test_triage_escalates_when_the_model_is_unavailable(conn):
    """A provider outage must not lose the ticket."""
    outcome = agent.triage(conn, "tkt_9001", FailingLLM())
    assert outcome.should_escalate
    trace = traces.get(conn, outcome.run_id)
    assert trace["escalation_reason"]


def test_draft_is_grounded_and_cited(conn):
    agent.triage(conn, "tkt_9001", HeuristicLLM())
    outcome = agent.draft_reply(conn, "tkt_9001", HeuristicLLM())

    assert outcome.status == agent.STATUS_DRAFTED
    assert outcome.citations, "a grounded reply must cite at least one policy"
    assert all(doc_id in outcome.retrieved_doc_ids for doc_id in outcome.citations)
    assert "KB-REFUND-001" in outcome.citations


def test_draft_mentions_the_customers_own_order(conn):
    """The reply uses this ticket's order, which is also the PII boundary."""
    agent.triage(conn, "tkt_9001", HeuristicLLM())
    outcome = agent.draft_reply(conn, "tkt_9001", HeuristicLLM())
    assert "ord_5001" in outcome.body
    # Another customer's identifiers must not appear.
    assert "cus_1006" not in outcome.body
    assert "ord_5006" not in outcome.body


def test_invented_citations_are_dropped(conn):
    """The model may not cite a document retrieval did not return."""
    agent.triage(conn, "tkt_9001", HeuristicLLM())
    scripted = ScriptedLLM([
        {"suspicious": False, "kinds": [], "reason": "clean"},
        {"body": "Our policy allows a full refund immediately.",
         "citations": ["KB-REFUND-001", "KB-TOTALLY-INVENTED-001"]},
    ])
    outcome = agent.draft_reply(conn, "tkt_9001", scripted)
    assert "KB-TOTALLY-INVENTED-001" not in outcome.citations


def test_draft_without_any_valid_citation_is_not_presented_as_policy(conn):
    agent.triage(conn, "tkt_9001", HeuristicLLM())
    scripted = ScriptedLLM([
        {"suspicious": False, "kinds": [], "reason": "clean"},
        {"body": "Sure, refunded already.", "citations": ["KB-MADE-UP-001"]},
    ])
    outcome = agent.draft_reply(conn, "tkt_9001", scripted)
    assert outcome.status != agent.STATUS_DRAFTED
    assert outcome.citations == []
    assert outcome.escalation_reason


def test_draft_escalates_when_the_model_fails(conn):
    agent.triage(conn, "tkt_9001", HeuristicLLM())
    outcome = agent.draft_reply(conn, "tkt_9001", FailingLLM())
    assert outcome.status == agent.STATUS_ESCALATED
    assert "unavailable" in (outcome.escalation_reason or "")
    assert outcome.body, "the customer still gets a holding reply"


def test_uncovered_question_asks_instead_of_guessing(conn):
    """No supporting policy means no invented answer."""
    conn.execute(
        """INSERT INTO tickets (ticket_id, customer_id, order_id, channel, subject, body,
                                created_at, status)
           VALUES ('tkt_off', 'cus_1001', 'ord_5001', 'email',
                   'Question', 'What is the weather in Paris tomorrow?',
                   '2026-07-02T10:00:00+05:30', 'open')"""
    )
    agent.triage(conn, "tkt_off", HeuristicLLM())
    outcome = agent.draft_reply(conn, "tkt_off", HeuristicLLM())

    assert outcome.status == agent.STATUS_NEEDS_INFO
    assert outcome.escalation_reason
    assert outcome.citations == []


def test_escalation_reasons_surface_as_policy_gaps(conn):
    """Logged escalation reasons double as the missing-policy backlog."""
    outcome = agent.triage(conn, "tkt_9004", HeuristicLLM())
    assert outcome.should_escalate

    gaps = traces.escalation_gaps(conn)
    assert any(gap["ticket_id"] == "tkt_9004" for gap in gaps)


def test_final_sale_item_gets_no_action_and_no_escalation(conn):
    """eval_003: policy says final sale, so the correct answer is a polite no."""
    triage_outcome = agent.triage(conn, "tkt_9003", HeuristicLLM())
    draft_outcome = agent.draft_reply(conn, "tkt_9003", HeuristicLLM())

    assert triage_outcome.category == "refund"
    assert not triage_outcome.should_escalate
    assert draft_outcome.recommended_actions == []


def test_return_window_is_judged_from_the_ticket_date_not_today(conn):
    """tkt_9001 was raised inside its window, so a replacement is allowed."""
    from app.repository import get_ticket_context

    context = get_ticket_context(conn, "tkt_9001")
    actions = agent.recommend_actions(
        context, "refund", should_escalate=False, guardrail_unsafe=False, covered=True
    )
    assert [a["tool_name"] for a in actions] == ["create_replacement_order"]

    # Same ticket, but pretend it was raised long after the window closed.
    context["ticket"]["created_at"] = "2027-01-01T00:00:00+05:30"
    later = agent.recommend_actions(
        context, "refund", should_escalate=False, guardrail_unsafe=False, covered=True
    )
    assert [a["tool_name"] for a in later] == ["escalate_to_human"]


def test_duplicate_citations_from_the_model_are_collapsed(conn):
    agent.triage(conn, "tkt_9001", HeuristicLLM())
    scripted = ScriptedLLM([
        {"suspicious": False, "kinds": [], "reason": "clean"},
        {"body": "Damaged items can be replaced.",
         "citations": ["KB-REFUND-001", "KB-REFUND-001"]},
    ])
    outcome = agent.draft_reply(conn, "tkt_9001", scripted)
    assert outcome.citations == ["KB-REFUND-001"]


def test_no_ticket_cites_the_same_document_twice(conn):
    for row in conn.execute("SELECT ticket_id FROM tickets").fetchall():
        agent.triage(conn, row["ticket_id"], HeuristicLLM())
        citations = agent.draft_reply(conn, row["ticket_id"], HeuristicLLM()).citations
        assert len(citations) == len(set(citations)), (row["ticket_id"], citations)
