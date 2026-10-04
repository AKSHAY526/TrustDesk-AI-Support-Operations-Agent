"""Hybrid retrieval over chunked policy documents."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from .config import settings
from .db import fts_available, get_meta
from .embeddings import from_blob, get_embedder

_WORD_RE = re.compile(r"[A-Za-z0-9\-]+")

# Semantic generalises better; keyword is what catches exact ids like KB-REFUND-001.
WEIGHT_SEMANTIC = 0.6
WEIGHT_KEYWORD = 0.4

# BM25 magnitudes on this corpus run roughly 1-8; m / (m + 3) spreads them over (0, 1).
BM25_REFERENCE = 3.0

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "my", "me", "i", "is", "are", "was", "were", "be", "been", "it",
    "this", "that", "please", "hi", "hello", "you", "your", "can", "could",
    "would", "do", "does", "did", "have", "has", "had", "not", "no", "so",
    "get", "got", "need", "want", "any", "all", "some", "there", "here",
    "about", "from", "they", "them", "then", "than", "what", "when", "how",
    "will", "just", "also", "been", "being", "much", "very", "still",
}


@dataclass
class RetrievedChunk:
    """One policy section returned by search."""

    chunk_id: str
    doc_id: str
    heading: str
    text: str
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    score: float = 0.0
    channels: list[str] = field(default_factory=list)

    @property
    def snippet(self) -> str:
        flat = " ".join(self.text.split())
        return flat[:220] + ("..." if len(flat) > 220 else "")

    def to_result(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "heading": self.heading,
            "snippet": self.snippet,
            "score": round(self.score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "keyword_score": round(self.keyword_score, 4),
            "channels": self.channels,
        }


def _fts_query(text: str) -> str:
    """Build an FTS5 OR query. Terms are quoted, so operators in ticket text do nothing."""
    terms = [t.lower() for t in _WORD_RE.findall(text)]
    terms = [t for t in terms if len(t) > 2 and t not in _STOPWORDS]
    # Keep the query bounded; the longest tickets are mostly prose.
    unique: list[str] = []
    for term in terms:
        if term not in unique:
            unique.append(term)
    return " OR ".join(f'"{t}"' for t in unique[:32])


def _keyword_search(conn: sqlite3.Connection, query: str, limit: int) -> dict[str, float]:
    """Return ``{chunk_id: score}`` in (0, 1], derived from BM25."""
    if not fts_available(conn):
        return {}
    match = _fts_query(query)
    if not match:
        return {}
    try:
        # Column weights: chunk_id, doc_id, heading (low, so a short section can't
        # win on its title alone), text.
        rows = conn.execute(
            """SELECT chunk_id, bm25(doc_chunks_fts, 0.0, 0.0, 0.4, 1.0) AS rank_score
               FROM doc_chunks_fts
               WHERE doc_chunks_fts MATCH ?
               ORDER BY rank_score LIMIT ?""",
            (match, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}

    # bm25() is negative, lower being better. Use a fixed scale, not one relative
    # to the best hit: relative scaling gives a lone weak match 1.0, which made
    # off-topic questions look covered.
    scores: dict[str, float] = {}
    for row in rows:
        magnitude = -float(row["rank_score"])
        if magnitude > 0:
            scores[row["chunk_id"]] = magnitude / (magnitude + BM25_REFERENCE)
    return scores


class StaleEmbeddingsError(RuntimeError):
    """Stored vectors were written by a different embedder than the active one."""


def _semantic_search(conn: sqlite3.Connection, query: str) -> dict[str, float]:
    """Return ``{chunk_id: cosine}`` for every chunk, clamped to >= 0."""
    rows = conn.execute("SELECT chunk_id, embedding FROM doc_chunks").fetchall()
    if not rows:
        return {}

    embedder = get_embedder()
    stored_name = get_meta(conn, "embedder_name")
    if stored_name is not None and stored_name != embedder.name:
        # Fail loudly; silently scoring zero would look like a quality drop.
        raise StaleEmbeddingsError(
            f"stored embeddings were written by {stored_name!r} but the active "
            f"embedder is {embedder.name!r}. Re-index with: "
            f"python -m scripts.seed --reset"
        )

    query_vector = embedder.encode([query])[0]
    scores: dict[str, float] = {}
    for row in rows:
        stored = from_blob(row["embedding"], query_vector.shape[0])
        if stored.shape != query_vector.shape:
            raise StaleEmbeddingsError(
                f"stored vector has {stored.shape[0]} dimensions but the active "
                f"embedder produces {query_vector.shape[0]}. Re-index with: "
                f"python -m scripts.seed --reset"
            )
        scores[row["chunk_id"]] = float(max(0.0, np.dot(stored, query_vector)))
    return scores


def search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Hybrid search. Returns chunks ordered by fused score, best first."""
    top_k = top_k or settings.retrieval_top_k
    if not query or not query.strip():
        return []

    pool = max(top_k * 4, 12)
    keyword = _keyword_search(conn, query, pool)
    semantic = _semantic_search(conn, query)

    candidate_ids = set(keyword) | {
        cid for cid, _ in sorted(semantic.items(), key=lambda kv: kv[1], reverse=True)[:pool]
    }
    if not candidate_ids:
        return []

    placeholders = ",".join("?" for _ in candidate_ids)
    rows = conn.execute(
        f"SELECT chunk_id, doc_id, heading, text FROM doc_chunks WHERE chunk_id IN ({placeholders})",
        tuple(candidate_ids),
    ).fetchall()

    results: list[RetrievedChunk] = []
    for row in rows:
        chunk_id = row["chunk_id"]
        sem = semantic.get(chunk_id, 0.0)
        kw = keyword.get(chunk_id, 0.0)
        channels = []
        if kw > 0:
            channels.append("keyword")
        if sem > 0:
            channels.append("semantic")
        results.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=row["doc_id"],
                heading=row["heading"] or "",
                text=row["text"],
                semantic_score=sem,
                keyword_score=kw,
                score=WEIGHT_SEMANTIC * sem + WEIGHT_KEYWORD * kw,
                channels=channels,
            )
        )

    results.sort(key=lambda c: c.score, reverse=True)
    return results[:top_k]


def is_covered(chunks: list[RetrievedChunk], min_score: float | None = None) -> bool:
    """True when at least one section is a strong enough match to answer from."""
    threshold = settings.retrieval_min_score if min_score is None else min_score
    return bool(chunks) and chunks[0].score >= threshold


def unique_doc_ids(chunks: list[RetrievedChunk]) -> list[str]:
    """Document ids in rank order, de-duplicated."""
    seen: list[str] = []
    for chunk in chunks:
        if chunk.doc_id not in seen:
            seen.append(chunk.doc_id)
    return seen
