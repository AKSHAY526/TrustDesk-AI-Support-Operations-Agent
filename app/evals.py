"""Evaluation runner."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from . import agent, traces
from .llm import LLMAdapter, get_llm


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ratio(hits: int, total: int) -> float:
    return round(hits / total, 4) if total else 1.0


def load_cases(conn: sqlite3.Connection, case_ids: list[str] | None = None) -> list[dict]:
    sql = "SELECT case_id, ticket_id, input, expected FROM eval_cases"
    params: tuple = ()
    if case_ids:
        placeholders = ",".join("?" for _ in case_ids)
        sql += f" WHERE case_id IN ({placeholders})"
        params = tuple(case_ids)
    sql += " ORDER BY case_id"
    return [
        {
            "case_id": row["case_id"],
            "ticket_id": row["ticket_id"],
            "input": row["input"],
            "expected": json.loads(row["expected"] or "{}"),
        }
        for row in conn.execute(sql, params)
    ]


def run_case(conn: sqlite3.Connection, case: dict, llm: LLMAdapter) -> dict:
    """Run one case and score it."""
    expected = case["expected"]
    ticket_id = case["ticket_id"]

    triage_outcome = agent.triage(conn, ticket_id, llm)
    draft_outcome = agent.draft_reply(conn, ticket_id, llm)

    if triage_outcome is None or draft_outcome is None:
        return {
            "case_id": case["case_id"],
            "ticket_id": ticket_id,
            "passed": False,
            "failures": ["ticket_not_found"],
        }

    recommended = [action["tool_name"] for action in draft_outcome.recommended_actions]
    citations = draft_outcome.citations

    must_cite = expected.get("must_cite_doc_ids", []) or []
    allowed_actions = expected.get("allowed_actions", []) or []
    disallowed = expected.get("disallowed_actions", []) or []

    category_ok = triage_outcome.category == expected.get("category")
    priority_ok = triage_outcome.priority == expected.get("priority")
    citation_ok = all(doc_id in citations for doc_id in must_cite)
    blocked_ok = not any(tool in disallowed for tool in recommended)
    escalation_ok = triage_outcome.should_escalate == bool(expected.get("should_escalate"))
    # Every recommendation must be one the case permits.
    action_ok = all(tool in allowed_actions for tool in recommended)

    failures = []
    if not category_ok:
        failures.append(f"category={triage_outcome.category} expected={expected.get('category')}")
    if not priority_ok:
        failures.append(f"priority={triage_outcome.priority} expected={expected.get('priority')}")
    if not citation_ok:
        failures.append(f"citations={citations} missing={[d for d in must_cite if d not in citations]}")
    if not blocked_ok:
        failures.append(f"recommended a disallowed action: {[t for t in recommended if t in disallowed]}")
    if not escalation_ok:
        failures.append(
            f"should_escalate={triage_outcome.should_escalate} expected={expected.get('should_escalate')}"
        )
    if not action_ok:
        failures.append(f"actions={recommended} allowed={allowed_actions}")

    traces.record(
        conn,
        run_id=traces.new_run_id(),
        ticket_id=ticket_id,
        run_type=traces.RUN_EVAL,
        status="passed" if not failures else "failed",
        retrieved_doc_ids=draft_outcome.retrieved_doc_ids,
        tool_calls=[{"tool_name": t} for t in recommended],
        guardrail_results=draft_outcome.guardrail,
        escalation_reason=draft_outcome.escalation_reason,
        model_name=draft_outcome.model,
    )

    return {
        "case_id": case["case_id"],
        "ticket_id": ticket_id,
        "passed": not failures,
        "failures": failures,
        "category": triage_outcome.category,
        "priority": triage_outcome.priority,
        "sentiment": triage_outcome.sentiment,
        "should_escalate": triage_outcome.should_escalate,
        "citations": citations,
        "recommended_actions": recommended,
        "draft_status": draft_outcome.status,
        "scores": {
            "category_ok": category_ok,
            "priority_ok": priority_ok,
            "citation_ok": citation_ok,
            "unsafe_blocked": blocked_ok,
            "escalation_ok": escalation_ok,
            "actions_allowed": action_ok,
        },
    }


def run(
    conn: sqlite3.Connection,
    case_ids: list[str] | None = None,
    llm: LLMAdapter | None = None,
) -> dict:
    """Run the eval set and persist a summary. Returns the report."""
    llm = llm or get_llm()
    cases = load_cases(conn, case_ids)
    started = _now()
    results = [run_case(conn, case, llm) for case in cases]
    total = len(results)

    def count(key: str) -> int:
        return sum(1 for r in results if r.get("scores", {}).get(key))

    metrics = {
        "triage_accuracy": _ratio(
            sum(
                1
                for r in results
                if r.get("scores", {}).get("category_ok") and r.get("scores", {}).get("priority_ok")
            ),
            total,
        ),
        "category_accuracy": _ratio(count("category_ok"), total),
        "priority_accuracy": _ratio(count("priority_ok"), total),
        "citation_coverage": _ratio(count("citation_ok"), total),
        "unsafe_action_block_rate": _ratio(count("unsafe_blocked"), total),
        "escalation_accuracy": _ratio(count("escalation_ok"), total),
        "allowed_action_accuracy": _ratio(count("actions_allowed"), total),
        "cases_fully_passed": _ratio(sum(1 for r in results if r["passed"]), total),
    }

    report = {
        "eval_run_id": f"eval_{uuid.uuid4().hex[:10]}",
        "started_at": started,
        "completed_at": _now(),
        "total_cases": total,
        "model": getattr(llm, "name", "unknown"),
        "metrics": metrics,
        "case_results": results,
    }

    conn.execute(
        """INSERT INTO eval_runs
           (eval_run_id, started_at, completed_at, total_cases, model, metrics, case_results)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            report["eval_run_id"], report["started_at"], report["completed_at"],
            total, report["model"], json.dumps(metrics), json.dumps(results),
        ),
    )
    return report


def latest(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute(
        "SELECT * FROM eval_runs ORDER BY completed_at DESC, rowid DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    report = dict(row)
    report["metrics"] = json.loads(report.get("metrics") or "{}")
    report["case_results"] = json.loads(report.get("case_results") or "[]")
    report["model"] = report.get("model") or "unknown"
    return report


def format_report(report: dict) -> str:
    """Human-readable summary for the CLI."""
    lines = [
        "",
        "TrustDesk evaluation",
        "=" * 64,
        f"run     {report['eval_run_id']}",
        f"model   {report['model']}",
        f"cases   {report['total_cases']}",
        "",
        f"{'case':<11}{'result':<9}{'category':<18}{'priority':<10}detail",
        "-" * 64,
    ]
    for result in report["case_results"]:
        verdict = "PASS" if result["passed"] else "FAIL"
        detail = "" if result["passed"] else result["failures"][0]
        lines.append(
            f"{result['case_id']:<11}{verdict:<9}{result.get('category', '-'):<18}"
            f"{result.get('priority', '-'):<10}{detail}"
        )
    lines += ["", "Metrics", "-" * 64]
    for key, value in report["metrics"].items():
        lines.append(f"  {key:<28}{value:.2%}")
    lines.append("")
    return "\n".join(lines)
