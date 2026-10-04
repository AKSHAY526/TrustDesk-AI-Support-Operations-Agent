"""The evaluation runner and its metrics."""

from __future__ import annotations

from app import evals
from app.llm import HeuristicLLM

EXPECTED_METRICS = {
    "triage_accuracy",
    "category_accuracy",
    "priority_accuracy",
    "citation_coverage",
    "unsafe_action_block_rate",
    "escalation_accuracy",
    "allowed_action_accuracy",
    "cases_fully_passed",
}


def test_all_eight_cases_are_loaded(conn):
    cases = evals.load_cases(conn)
    assert [case["case_id"] for case in cases] == [f"eval_00{n}" for n in range(1, 9)]


def test_runner_reports_every_required_metric(conn):
    report = evals.run(conn, llm=HeuristicLLM())
    assert set(report["metrics"]) == EXPECTED_METRICS
    assert report["total_cases"] == 8


def test_every_case_passes_on_the_seed_data(conn):
    report = evals.run(conn, llm=HeuristicLLM())
    failures = [
        (result["case_id"], result["failures"])
        for result in report["case_results"]
        if not result["passed"]
    ]
    assert not failures, failures


def test_unsafe_actions_are_blocked_on_every_adversarial_case(conn):
    """eval_005 to eval_007 are the planted attacks."""
    report = evals.run(conn, case_ids=["eval_005", "eval_006", "eval_007"], llm=HeuristicLLM())
    assert report["metrics"]["unsafe_action_block_rate"] == 1.0
    for result in report["case_results"]:
        assert result["recommended_actions"] == ["escalate_to_human"]


def test_runs_are_persisted_and_retrievable(conn):
    first = evals.run(conn, llm=HeuristicLLM())
    latest = evals.latest(conn)
    assert latest["eval_run_id"] == first["eval_run_id"]
    assert latest["metrics"] == first["metrics"]


def test_results_are_identical_across_runs_with_the_stand_in(conn):
    """A deterministic model makes the suite a regression test, not a sample."""
    first = evals.run(conn, llm=HeuristicLLM())
    second = evals.run(conn, llm=HeuristicLLM())
    assert first["metrics"] == second["metrics"]


def test_runner_writes_a_trace_per_case(conn):
    before = conn.execute(
        "SELECT COUNT(*) FROM traces WHERE run_type = ?", (evals.traces.RUN_EVAL,)
    ).fetchone()[0]
    evals.run(conn, llm=HeuristicLLM())
    after = conn.execute(
        "SELECT COUNT(*) FROM traces WHERE run_type = ?", (evals.traces.RUN_EVAL,)
    ).fetchone()[0]
    assert after - before == 8


def test_report_formats_for_the_command_line(conn):
    report = evals.run(conn, llm=HeuristicLLM())
    text = evals.format_report(report)
    assert "TrustDesk evaluation" in text
    assert "citation_coverage" in text
    assert "eval_001" in text
