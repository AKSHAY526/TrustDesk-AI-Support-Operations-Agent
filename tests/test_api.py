"""End-to-end API behaviour, including roles and the approval gate."""

from __future__ import annotations

from tests.conftest import ADMIN, AGENT, MANAGER


def test_health_reports_active_configuration(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "model" in body and "embedder" in body


def test_endpoints_require_a_token(client):
    for method, path in [
        ("get", "/tickets"),
        ("get", "/tickets/tkt_9001"),
        ("get", "/kb/search?q=refund"),
        ("post", "/tickets/tkt_9001/triage"),
        ("get", "/actions/catalogue"),
    ]:
        response = getattr(client, method)(path)
        assert response.status_code == 401, path


def test_invalid_token_is_rejected(client):
    response = client.get("/tickets", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401
    assert response.json()["detail"]["error"] == "invalid_token"


def test_list_and_open_a_ticket(client):
    tickets = client.get("/tickets", headers=AGENT).json()
    assert len(tickets) == 8

    detail = client.get("/tickets/tkt_9001", headers=AGENT).json()
    assert detail["ticket"]["ticket_id"] == "tkt_9001"
    assert detail["customer"]["customer_id"] == "cus_1001"
    assert detail["order"]["order_id"] == "ord_5001"


def test_ticket_response_hides_the_evaluation_labels(client):
    """Production responses must not leak the answer key."""
    detail = client.get("/tickets/tkt_9001", headers=AGENT).json()
    assert "expected_category" not in detail["ticket"]
    assert "expected_actions" not in detail["ticket"]


def test_missing_ticket_returns_a_structured_error(client):
    response = client.get("/tickets/tkt_does_not_exist", headers=AGENT)
    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "ticket_not_found"


def test_search_reports_coverage_and_threshold(client):
    body = client.get("/kb/search", params={"q": "damaged item replacement"}, headers=AGENT).json()
    assert body["covered"] is True
    assert body["threshold"] > 0
    assert body["results"][0]["doc_id"].startswith("KB-")


def test_triage_then_draft_produces_citations(client):
    triage = client.post("/tickets/tkt_9001/triage", headers=AGENT).json()
    assert triage["category"] == "refund"

    draft = client.post("/tickets/tkt_9001/draft", headers=AGENT).json()
    assert draft["citations"]
    assert all(doc in draft["retrieved_doc_ids"] for doc in draft["citations"])
    assert [a["tool_name"] for a in draft["recommended_actions"]] == ["create_replacement_order"]


def test_draft_triages_first_when_needed(client):
    """Drafting depends on the category, so it must not silently use a default."""
    draft = client.post("/tickets/tkt_9008/draft", headers=AGENT).json()
    assert [a["tool_name"] for a in draft["recommended_actions"]] == ["start_refund_review"]


def test_catalogue_is_exposed_with_its_rules(client):
    catalogue = {tool["tool_name"]: tool for tool in client.get("/actions/catalogue", headers=AGENT).json()}
    assert len(catalogue) == 6
    assert catalogue["issue_coupon"]["max_amount_inr"] == 1000


def test_approval_gate_end_to_end(client):
    client.post("/tickets/tkt_9001/triage", headers=AGENT)
    draft = client.post("/tickets/tkt_9001/draft", headers=AGENT).json()
    recommended = draft["recommended_actions"][0]

    created = client.post(
        "/actions",
        json={"ticket_id": "tkt_9001", "tool_name": recommended["tool_name"],
              "payload": recommended["payload"]},
        headers=AGENT,
    ).json()
    action_id = created["action"]["action_id"]
    assert created["action"]["status"] == "approval_required"
    assert created["action"]["result"] is None

    # An agent may not approve.
    forbidden = client.post(f"/actions/{action_id}/approve", json={"decision": "approved"}, headers=AGENT)
    assert forbidden.status_code == 403
    assert forbidden.json()["detail"]["error"] == "insufficient_role"

    approved = client.post(
        f"/actions/{action_id}/approve",
        json={"decision": "approved", "reason": "damage photo received"},
        headers=MANAGER,
    ).json()
    assert approved["executed"] is True
    assert approved["action"]["status"] == "executed"
    assert approved["action"]["result"]["replacement_order_id"]


def test_duplicate_request_is_idempotent_over_http(client):
    client.post("/tickets/tkt_9001/triage", headers=AGENT)
    draft = client.post("/tickets/tkt_9001/draft", headers=AGENT).json()
    recommended = draft["recommended_actions"][0]
    payload = {"ticket_id": "tkt_9001", "tool_name": recommended["tool_name"],
               "payload": recommended["payload"]}

    first = client.post("/actions", json=payload, headers=AGENT).json()
    second = client.post("/actions", json=payload, headers=AGENT).json()

    assert second["replayed"] is True
    assert second["action"]["action_id"] == first["action"]["action_id"]
    assert len(client.get("/actions", params={"ticket_id": "tkt_9001"}, headers=AGENT).json()) == 1


def test_approval_queue_is_for_managers_and_admins(client):
    draft = client.post("/tickets/tkt_9001/draft", headers=AGENT).json()
    recommended = draft["recommended_actions"][0]
    created = client.post(
        "/actions",
        json={"ticket_id": "tkt_9001", "tool_name": recommended["tool_name"],
              "payload": recommended["payload"]},
        headers=AGENT,
    ).json()
    action_id = created["action"]["action_id"]

    # An agent can follow their own ticket, but not browse every ticket's actions.
    assert client.get("/actions", headers=AGENT).status_code == 403
    mine = client.get("/actions", params={"ticket_id": "tkt_9001"}, headers=AGENT).json()
    assert [a["action_id"] for a in mine] == [action_id]

    queue = client.get("/actions", params={"status": "approval_required"}, headers=MANAGER).json()
    assert [a["action_id"] for a in queue] == [action_id]

    client.post(f"/actions/{action_id}/approve", json={"decision": "approved"}, headers=MANAGER)
    assert client.get("/actions", params={"status": "approval_required"}, headers=ADMIN).json() == []


def test_amount_cap_is_enforced_over_http(client):
    client.post("/tickets/tkt_9006/triage", headers=AGENT)
    response = client.post(
        "/actions",
        json={"ticket_id": "tkt_9006", "tool_name": "issue_coupon",
              "payload": {"customer_id": "cus_1006", "amount": 5000, "reason": "goodwill",
                          "idempotency_key": "http-coupon-1"}},
        headers=AGENT,
    )
    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "amount_over_limit"


def test_trace_is_retrievable_for_a_run(client):
    draft = client.post("/tickets/tkt_9001/draft", headers=AGENT).json()
    trace = client.get(f"/traces/{draft['run_id']}", headers=AGENT).json()
    assert trace["run_type"] == "draft_reply"
    assert trace["retrieved_doc_ids"]
    assert trace["prompt_version"]


def test_policy_gaps_list_escalation_reasons(client):
    client.post("/tickets/tkt_9004/triage", headers=AGENT)
    gaps = client.get("/insights/policy-gaps", headers=MANAGER).json()
    assert any(gap["ticket_id"] == "tkt_9004" and gap["escalation_reason"] for gap in gaps)


def test_policy_gaps_are_for_managers_and_admins(client):
    """The console hides this panel from agents, so the API refuses them too."""
    refused = client.get("/insights/policy-gaps", headers=AGENT)
    assert refused.status_code == 403
    assert refused.json()["detail"]["error"] == "insufficient_role"
    assert client.get("/insights/policy-gaps", headers=MANAGER).status_code == 200
    assert client.get("/insights/policy-gaps", headers=ADMIN).status_code == 200


def test_only_admin_may_run_evaluations(client):
    assert client.post("/evals/run", headers=AGENT).status_code == 403
    assert client.post("/evals/run", headers=MANAGER).status_code == 403

    report = client.post("/evals/run", headers=ADMIN).json()
    assert report["total_cases"] == 8
    assert report["metrics"]["unsafe_action_block_rate"] == 1.0


def test_latest_eval_is_available_after_a_run(client):
    client.post("/evals/run", headers=ADMIN)
    latest = client.get("/evals/latest", headers=ADMIN).json()
    assert latest["total_cases"] == 8


def test_eval_results_are_for_admins_only(client):
    client.post("/evals/run", headers=ADMIN)
    assert client.get("/evals/latest", headers=AGENT).status_code == 403
    assert client.get("/evals/latest", headers=MANAGER).status_code == 403
    assert client.get("/evals/latest", headers=ADMIN).status_code == 200


def test_only_admin_may_ingest_documents(client):
    document = {"documents": [{"doc_id": "KB-TEST-001", "title": "Test Policy",
                               "content": "# Test Policy\n## Rule\nTest widgets may be returned."}]}
    assert client.post("/kb/ingest", json=document, headers=AGENT).status_code == 403

    result = client.post("/kb/ingest", json=document, headers=ADMIN).json()
    assert result["ingested"] == 1
    assert result["document_ids"] == ["KB-TEST-001"]

    found = client.get("/kb/search", params={"q": "test widgets returned"}, headers=ADMIN).json()
    assert any(r["doc_id"] == "KB-TEST-001" for r in found["results"])


def test_openapi_schema_is_generated(client):
    """The API documentation deliverable comes from this."""
    schema = client.get("/openapi.json").json()
    assert "/tickets/{ticket_id}/triage" in schema["paths"]
    assert "/actions/{action_id}/approve" in schema["paths"]
