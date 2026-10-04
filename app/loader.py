"""Load the provided seed data into SQLite."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA_DIR, KB_DIR
from .db import fts_available, init_db, set_meta
from .embeddings import get_embedder, to_blob

_DOC_ID_RE = re.compile(r"^Doc ID:\s*(?P<value>\S+)", re.MULTILINE)
_AUDIENCE_RE = re.compile(r"^Audience:\s*(?P<value>.+)$", re.MULTILINE)
_VERSION_RE = re.compile(r"^Version:\s*(?P<value>.+)$", re.MULTILINE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_kb_document(path: Path) -> dict:
    """Parse one knowledge-base Markdown file into metadata plus sections."""
    raw = path.read_text(encoding="utf-8")

    title_match = re.search(r"^#\s+(?P<value>.+)$", raw, re.MULTILINE)
    doc_id_match = _DOC_ID_RE.search(raw)
    if not doc_id_match:
        raise ValueError(f"{path.name} has no 'Doc ID:' line")

    audience_match = _AUDIENCE_RE.search(raw)
    version_match = _VERSION_RE.search(raw)

    # Split on "## " headings. Anything before the first heading is metadata.
    parts = re.split(r"^##\s+(.+)$", raw, flags=re.MULTILINE)
    sections: list[tuple[str, str]] = []
    if len(parts) > 1:
        # parts = [preamble, heading1, body1, heading2, body2, ...]
        for heading, bodytext in zip(parts[1::2], parts[2::2]):
            cleaned = bodytext.strip()
            if cleaned:
                sections.append((heading.strip(), cleaned))
    if not sections:
        # No sections: treat the whole document (minus metadata lines) as one.
        body = re.sub(r"^(#\s+.+|Doc ID:.*|Audience:.*|Version:.*)$", "", raw, flags=re.MULTILINE)
        sections = [("", body.strip())]

    return {
        "doc_id": doc_id_match.group("value").strip(),
        "title": title_match.group("value").strip() if title_match else path.stem,
        "audience": audience_match.group("value").strip() if audience_match else "",
        "version": version_match.group("value").strip() if version_match else "",
        "source_path": f"data/knowledge_base/{path.name}",
        "content": raw,
        "checksum": hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16],
        "sections": sections,
    }


def _load_json(name: str) -> list[dict]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def _load_jsonl(name: str) -> list[dict]:
    lines = (DATA_DIR / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_customers(conn: sqlite3.Connection) -> int:
    rows = _load_json("customers.json")
    conn.executemany(
        """INSERT OR REPLACE INTO customers
           (customer_id, name, email, tier, country, created_at, verified, tags)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                r["customer_id"], r.get("name"), r.get("email"), r.get("tier"),
                r.get("country"), r.get("created_at"), int(bool(r.get("verified"))),
                json.dumps(r.get("tags", [])),
            )
            for r in rows
        ],
    )
    return len(rows)


def load_orders(conn: sqlite3.Connection) -> int:
    rows = _load_json("orders.json")
    conn.executemany(
        """INSERT OR REPLACE INTO orders
           (order_id, customer_id, status, placed_at, delivered_at,
            eligible_return_until, total, currency, payment_status,
            tracking_number, items)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                r["order_id"], r.get("customer_id"), r.get("status"), r.get("placed_at"),
                r.get("delivered_at"), r.get("eligible_return_until"), r.get("total"),
                r.get("currency"), r.get("payment_status"), r.get("tracking_number"),
                json.dumps(r.get("items", [])),
            )
            for r in rows
        ],
    )
    return len(rows)


def load_tickets(conn: sqlite3.Connection) -> int:
    rows = _load_json("tickets.json")
    conn.executemany(
        """INSERT OR REPLACE INTO tickets
           (ticket_id, customer_id, order_id, channel, subject, body, created_at,
            status, expected_category, expected_priority, expected_sentiment,
            expected_escalation, expected_actions)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                r["ticket_id"], r.get("customer_id"), r.get("order_id"), r.get("channel"),
                r.get("subject"), r.get("body"), r.get("created_at"), r.get("status"),
                r.get("expected_category"), r.get("expected_priority"),
                r.get("expected_sentiment"),
                None if r.get("expected_escalation") is None else int(bool(r["expected_escalation"])),
                json.dumps(r.get("expected_actions", [])),
            )
            for r in rows
        ],
    )
    return len(rows)


def load_tool_catalog(conn: sqlite3.Connection) -> int:
    rows = _load_json("tool_actions.json")
    conn.executemany(
        """INSERT OR REPLACE INTO tool_catalog
           (tool_name, description, risk_level, requires_human_approval,
            allowed_categories, required_fields, max_amount_inr)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                r["tool_name"], r.get("description"), r.get("risk_level"),
                int(bool(r.get("requires_human_approval"))),
                json.dumps(r.get("allowed_categories", [])),
                json.dumps(r.get("required_fields", [])),
                r.get("max_amount_inr"),
            )
            for r in rows
        ],
    )
    return len(rows)


def load_eval_cases(conn: sqlite3.Connection) -> int:
    rows = _load_jsonl("eval_cases.jsonl")
    conn.executemany(
        "INSERT OR REPLACE INTO eval_cases (case_id, ticket_id, input, expected) VALUES (?, ?, ?, ?)",
        [
            (r["case_id"], r.get("ticket_id"), r.get("input"), json.dumps(r.get("expected", {})))
            for r in rows
        ],
    )
    return len(rows)


def load_knowledge_base(conn: sqlite3.Connection) -> tuple[int, int]:
    """Load, chunk, index and embed the knowledge base. Returns (docs, chunks)."""
    docs = [parse_kb_document(p) for p in sorted(KB_DIR.glob("*.md"))]

    conn.executemany(
        """INSERT OR REPLACE INTO kb_documents
           (doc_id, title, audience, version, source_path, content, checksum, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                d["doc_id"], d["title"], d["audience"], d["version"],
                d["source_path"], d["content"], d["checksum"], _now(),
            )
            for d in docs
        ],
    )

    chunks: list[tuple[str, str, str, str, str]] = []
    for doc in docs:
        for index, (heading, text) in enumerate(doc["sections"], start=1):
            chunk_id = f"{doc['doc_id']}#s{index}"
            # Prepend title and heading so the embedded text carries context.
            embed_text = " - ".join(filter(None, [doc["title"], heading, text]))
            chunks.append((chunk_id, doc["doc_id"], heading, text, embed_text))

    embedder = get_embedder()
    vectors = embedder.encode([c[4] for c in chunks]) if chunks else []

    conn.execute("DELETE FROM doc_chunks")
    conn.executemany(
        "INSERT INTO doc_chunks (chunk_id, doc_id, heading, text, embedding) VALUES (?, ?, ?, ?, ?)",
        [
            (c[0], c[1], c[2], c[3], to_blob(vectors[i]))
            for i, c in enumerate(chunks)
        ],
    )

    if fts_available(conn):
        conn.execute("DELETE FROM doc_chunks_fts")
        conn.executemany(
            "INSERT INTO doc_chunks_fts (chunk_id, doc_id, heading, text) VALUES (?, ?, ?, ?)",
            [(c[0], c[1], c[2], f"{c[1]} {c[2]} {c[3]}") for c in chunks],
        )

    # Lets retrieval detect vectors written by a different embedder.
    set_meta(conn, "embedder_name", embedder.name)
    set_meta(conn, "embedder_dim", str(getattr(embedder, "dim", 0)))

    return len(docs), len(chunks)


def load_all(db_path: Path | str | None = None) -> dict:
    """Load every seed file. Returns a summary suitable for printing."""
    conn = init_db(db_path)
    try:
        conn.execute("BEGIN")
        summary = {
            "customers": load_customers(conn),
            "orders": load_orders(conn),
            "tickets": load_tickets(conn),
            "tools": load_tool_catalog(conn),
            "eval_cases": load_eval_cases(conn),
        }
        docs, chunks = load_knowledge_base(conn)
        summary["kb_documents"] = docs
        summary["kb_chunks"] = chunks
        summary["embedder"] = get_embedder().name
        conn.execute("COMMIT")
    except Exception:
        # The failure may already have ended the transaction.
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return summary
