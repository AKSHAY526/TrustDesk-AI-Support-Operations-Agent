"""Request and response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    error: str
    message: str


class TicketSummary(BaseModel):
    ticket_id: str
    customer_id: str | None = None
    order_id: str | None = None
    channel: str | None = None
    subject: str | None = None
    created_at: str | None = None
    status: str | None = None


class TicketContext(BaseModel):
    ticket: dict[str, Any]
    customer: dict[str, Any] | None = None
    order: dict[str, Any] | None = None
    triage: dict[str, Any] | None = None
    drafts: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)


class SearchResult(BaseModel):
    doc_id: str
    chunk_id: str
    heading: str
    snippet: str
    score: float
    semantic_score: float
    keyword_score: float
    channels: list[str]


class SearchResponse(BaseModel):
    query: str
    covered: bool = Field(description="True when a section cleared the relevance threshold.")
    threshold: float
    results: list[SearchResult]


class IngestDocument(BaseModel):
    doc_id: str
    title: str | None = None
    content: str
    source_path: str | None = None
    audience: str | None = None
    version: str | None = None


class IngestRequest(BaseModel):
    documents: list[IngestDocument]


class IngestResponse(BaseModel):
    ingested: int
    document_ids: list[str]
    chunks: int


class TriageResponse(BaseModel):
    ticket_id: str
    category: str
    priority: str
    sentiment: str
    should_escalate: bool
    reason_summary: str
    run_id: str
    retrieved_doc_ids: list[str]
    guardrail: dict[str, Any]
    model: str


class RecommendedAction(BaseModel):
    tool_name: str
    payload: dict[str, Any]
    reason: str


class DraftResponse(BaseModel):
    ticket_id: str
    draft_id: str
    status: str
    body: str
    citations: list[str]
    recommended_actions: list[RecommendedAction]
    run_id: str
    retrieved_doc_ids: list[str]
    guardrail: dict[str, Any]
    escalation_reason: str | None = None
    model: str


class ActionRequest(BaseModel):
    ticket_id: str
    tool_name: str
    payload: dict[str, Any] = Field(
        description="Must include every required field from the catalogue, including idempotency_key."
    )


class ActionDecision(BaseModel):
    decision: Literal["approved", "rejected"]
    reason: str | None = None


class ActionResponse(BaseModel):
    action: dict[str, Any]
    created: bool = False
    executed: bool = False
    replayed: bool = Field(
        default=False,
        description="True when the idempotency key was already seen; no new side effect occurred.",
    )


class EvalRunResponse(BaseModel):
    eval_run_id: str
    started_at: str
    completed_at: str
    total_cases: int
    model: str
    metrics: dict[str, float]
    case_results: list[dict[str, Any]]


class EvalQueuedResponse(BaseModel):
    status: Literal["queued"]
    message: str
