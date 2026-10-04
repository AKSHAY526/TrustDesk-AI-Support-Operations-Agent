"""Knowledge-base endpoints: hybrid search and admin ingestion."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query

from ..auth import Principal, current_principal, require_admin
from ..config import settings
from ..db import fts_available
from ..deps import get_db
from ..embeddings import get_embedder, to_blob
from ..models import IngestRequest, IngestResponse, SearchResponse
from ..retrieval import is_covered, search

router = APIRouter(prefix="/kb", tags=["knowledge base"])


@router.get("/search", response_model=SearchResponse, summary="Hybrid policy search")
def search_kb(
    q: str = Query(min_length=1, description="Natural-language query"),
    top_k: int | None = Query(default=None, ge=1, le=20),
    conn: sqlite3.Connection = Depends(get_db),
    principal: Principal = Depends(current_principal),
) -> dict:
    chunks = search(conn, q, top_k)
    return {
        "query": q,
        "covered": is_covered(chunks),
        "threshold": settings.retrieval_min_score,
        "results": [chunk.to_result() for chunk in chunks],
    }


@router.post("/ingest", response_model=IngestResponse, summary="Ingest documents (admin)")
def ingest(
    payload: IngestRequest,
    conn: sqlite3.Connection = Depends(get_db),
    principal: Principal = Depends(require_admin),
) -> dict:
    """Add or replace documents, preserving the supplied document ids."""
    embedder = get_embedder()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ingested: list[str] = []
    chunk_count = 0

    conn.execute("BEGIN")
    try:
        for document in payload.documents:
            conn.execute(
                """INSERT OR REPLACE INTO kb_documents
                   (doc_id, title, audience, version, source_path, content, checksum, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    document.doc_id, document.title or document.doc_id, document.audience or "",
                    document.version or "", document.source_path or "", document.content,
                    hashlib.sha256(document.content.encode("utf-8")).hexdigest()[:16], now,
                ),
            )

            # Chunk on "## " headings, matching how the seed loader splits files.
            sections: list[tuple[str, str]] = []
            current_heading = ""
            buffer: list[str] = []
            for line in document.content.splitlines():
                if line.startswith("## "):
                    if buffer:
                        sections.append((current_heading, "\n".join(buffer).strip()))
                        buffer = []
                    current_heading = line[3:].strip()
                else:
                    buffer.append(line)
            if buffer:
                sections.append((current_heading, "\n".join(buffer).strip()))
            sections = [(h, t) for h, t in sections if t]
            if not sections:
                sections = [("", document.content.strip())]

            conn.execute("DELETE FROM doc_chunks WHERE doc_id = ?", (document.doc_id,))
            if fts_available(conn):
                conn.execute("DELETE FROM doc_chunks_fts WHERE doc_id = ?", (document.doc_id,))

            title = document.title or document.doc_id
            texts = [" - ".join(filter(None, [title, heading, text])) for heading, text in sections]
            vectors = embedder.encode(texts)
            for index, (heading, text) in enumerate(sections, start=1):
                chunk_id = f"{document.doc_id}#s{index}"
                conn.execute(
                    """INSERT INTO doc_chunks (chunk_id, doc_id, heading, text, embedding)
                       VALUES (?, ?, ?, ?, ?)""",
                    (chunk_id, document.doc_id, heading, text, to_blob(vectors[index - 1])),
                )
                if fts_available(conn):
                    conn.execute(
                        """INSERT INTO doc_chunks_fts (chunk_id, doc_id, heading, text)
                           VALUES (?, ?, ?, ?)""",
                        (chunk_id, document.doc_id, heading, f"{document.doc_id} {heading} {text}"),
                    )
                chunk_count += 1

            ingested.append(document.doc_id)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    return {"ingested": len(ingested), "document_ids": ingested, "chunks": chunk_count}
