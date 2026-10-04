"""Seed loading, chunking and hybrid retrieval."""

from __future__ import annotations

import pytest

from app.config import settings
from app.retrieval import is_covered, search, unique_doc_ids


def test_seed_data_loads_expected_counts(conn):
    counts = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("customers", "orders", "tickets", "tool_catalog", "eval_cases", "kb_documents")
    }
    assert counts == {
        "customers": 6,
        "orders": 6,
        "tickets": 8,
        "tool_catalog": 6,
        "eval_cases": 8,
        "kb_documents": 8,
    }


def test_document_ids_are_preserved_exactly(conn):
    """The eval cases assert on these ids, so they must survive loading."""
    doc_ids = {row[0] for row in conn.execute("SELECT doc_id FROM kb_documents")}
    assert {
        "KB-REFUND-001", "KB-SHIPPING-001", "KB-WARRANTY-001", "KB-BILLING-001",
        "KB-ACCOUNT-001", "KB-SECURITY-001", "KB-COUPON-001", "KB-ADVERSARIAL-001",
    } <= doc_ids


def test_documents_are_split_into_sections(conn):
    """Chunking means we retrieve a clause, not a whole policy file."""
    total = conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0]
    refund = conn.execute(
        "SELECT COUNT(*) FROM doc_chunks WHERE doc_id = 'KB-REFUND-001'"
    ).fetchone()[0]
    assert total > 8, "expected more chunks than documents"
    assert refund >= 4, "refund policy has four headed sections"


def test_every_chunk_has_an_embedding(conn):
    missing = conn.execute(
        "SELECT COUNT(*) FROM doc_chunks WHERE embedding IS NULL OR length(embedding) = 0"
    ).fetchone()[0]
    assert missing == 0


@pytest.mark.parametrize(
    ("query", "expected_doc"),
    [
        ("tracking has shown no movement for six days", "KB-SHIPPING-001"),
        ("tablet battery is swelling after 13 months", "KB-WARRANTY-001"),
        ("I was charged twice for one order", "KB-BILLING-001"),
        ("change the email address on my account", "KB-ACCOUNT-001"),
        ("someone asked me to reveal the system prompt", "KB-SECURITY-001"),
    ],
)
def test_hybrid_search_ranks_the_governing_policy_first(conn, query, expected_doc):
    chunks = search(conn, query)
    assert is_covered(chunks)
    assert unique_doc_ids(chunks)[0] == expected_doc


def test_paraphrase_only_query_is_found_but_not_ranked_first_offline(conn):
    """No policy says "cracked", so the hashing fallback finds the refund policy but not first."""
    chunks = search(conn, "my earbuds arrived cracked, can I get a replacement")
    assert is_covered(chunks)
    assert "KB-REFUND-001" in unique_doc_ids(chunks)


def _sentence_transformers_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("sentence_transformers") is not None


@pytest.mark.skipif(
    not _sentence_transformers_installed(),
    reason="sentence-transformers is not installed; the offline fallback is used instead",
)
def test_semantic_model_separates_real_tickets_from_off_topic(tmp_path):
    """With real embeddings, one threshold separates real tickets from off-topic ones."""
    import json

    from app import embeddings as emb
    from app.config import DATA_DIR
    from app.db import connect as _connect
    from app.loader import load_all

    previous = emb.get_embedder()
    emb.set_embedder(emb.SentenceTransformerEmbedder())
    try:
        path = tmp_path / "semantic.db"
        load_all(path)
        connection = _connect(path)
        try:
            tickets = json.loads((DATA_DIR / "tickets.json").read_text(encoding="utf-8"))
            real = [
                search(connection, f"{t['subject']}\n{t['body']}")[0].score for t in tickets
            ]
            off_topic = [
                search(connection, q)[0].score
                for q in (
                    "What is the weather in Paris tomorrow?",
                    "Can you write a poem about the sea?",
                    "Who won the cricket match last night?",
                )
            ]
            assert min(real) > max(off_topic), (
                f"expected separation, got real min {min(real):.3f} "
                f"vs off-topic max {max(off_topic):.3f}"
            )
            assert min(real) >= settings.retrieval_min_score
            assert max(off_topic) < settings.retrieval_min_score
        finally:
            connection.close()
    finally:
        emb.set_embedder(previous)


def test_agent_retrieval_query_recovers_the_right_policy(conn):
    """The live path adds order context, which resolves the paraphrase case."""
    from app.agent import _retrieval_query
    from app.repository import get_ticket_context

    context = get_ticket_context(conn, "tkt_9001")
    chunks = search(conn, _retrieval_query(context))
    assert "KB-REFUND-001" in unique_doc_ids(chunks)[:2]


def test_unrelated_question_is_not_treated_as_covered(conn):
    """Without this the agent would answer from a loosely related policy."""
    chunks = search(conn, "what is the weather in Paris tomorrow")
    assert not is_covered(chunks)


def test_keyword_channel_matches_an_exact_document_id(conn):
    """Embeddings are weak at identifiers; the keyword channel carries them."""
    chunks = search(conn, "KB-WARRANTY-001")
    assert "KB-WARRANTY-001" in unique_doc_ids(chunks)
    assert any("keyword" in chunk.channels for chunk in chunks)


def test_semantic_channel_survives_rewording(conn):
    """No shared keyword with the policy text, yet it should still be found."""
    chunks = search(conn, "the thing I bought turned up smashed to pieces")
    assert chunks
    assert any(chunk.semantic_score > 0 for chunk in chunks)


def test_search_query_cannot_break_the_full_text_index(conn):
    """Customer text may contain FTS operators; they must be neutralised."""
    for hostile in ['refund" OR 1=1 --', "NEAR(a b", '"unbalanced', "*", "AND OR NOT"]:
        search(conn, hostile)  # must not raise


def test_threshold_is_configurable(conn):
    chunks = search(conn, "refund policy for damaged goods")
    assert is_covered(chunks, min_score=0.0)
    assert not is_covered(chunks, min_score=0.99)
    assert settings.retrieval_min_score > 0
