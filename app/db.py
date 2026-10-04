"""SQLite schema and connection handling."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator

from .config import DB_PATH

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT,
    tier        TEXT,
    country     TEXT,
    created_at  TEXT,
    verified    INTEGER DEFAULT 0,
    tags        TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    order_id              TEXT PRIMARY KEY,
    customer_id           TEXT NOT NULL,
    status                TEXT,
    placed_at             TEXT,
    delivered_at          TEXT,
    eligible_return_until TEXT,
    total                 REAL,
    currency              TEXT,
    payment_status        TEXT,
    tracking_number       TEXT,
    items                 TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

-- The customer's original message is never modified, for auditability.
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id   TEXT PRIMARY KEY,
    customer_id TEXT,
    order_id    TEXT,
    channel     TEXT,
    subject     TEXT,
    body        TEXT,
    created_at  TEXT,
    status      TEXT,
    -- Seed-only evaluation labels. Never read by production logic.
    expected_category   TEXT,
    expected_priority   TEXT,
    expected_sentiment  TEXT,
    expected_escalation INTEGER,
    expected_actions    TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

-- Records which embedder wrote the stored vectors.
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS kb_documents (
    doc_id      TEXT PRIMARY KEY,
    title       TEXT,
    audience    TEXT,
    version     TEXT,
    source_path TEXT,
    content     TEXT,
    checksum    TEXT,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS doc_chunks (
    chunk_id  TEXT PRIMARY KEY,
    doc_id    TEXT NOT NULL,
    heading   TEXT,
    text      TEXT NOT NULL,
    embedding BLOB,
    FOREIGN KEY (doc_id) REFERENCES kb_documents (doc_id)
);

-- Loaded from data/tool_actions.json. The tool engine reads its rules from here.
CREATE TABLE IF NOT EXISTS tool_catalog (
    tool_name               TEXT PRIMARY KEY,
    description             TEXT,
    risk_level              TEXT,
    requires_human_approval INTEGER,
    allowed_categories      TEXT,
    required_fields         TEXT,
    max_amount_inr          REAL
);

CREATE TABLE IF NOT EXISTS eval_cases (
    case_id   TEXT PRIMARY KEY,
    ticket_id TEXT,
    input     TEXT,
    expected  TEXT
);

CREATE TABLE IF NOT EXISTS triage_results (
    triage_id      TEXT PRIMARY KEY,
    ticket_id      TEXT NOT NULL,
    category       TEXT,
    priority       TEXT,
    sentiment      TEXT,
    should_escalate INTEGER,
    reason_summary TEXT,
    run_id         TEXT,
    created_at     TEXT,
    FOREIGN KEY (ticket_id) REFERENCES tickets (ticket_id)
);

CREATE TABLE IF NOT EXISTS drafts (
    draft_id   TEXT PRIMARY KEY,
    ticket_id  TEXT NOT NULL,
    status     TEXT,
    body       TEXT,
    citations  TEXT,
    run_id     TEXT,
    created_at TEXT,
    FOREIGN KEY (ticket_id) REFERENCES tickets (ticket_id)
);

CREATE TABLE IF NOT EXISTS tool_actions (
    action_id               TEXT PRIMARY KEY,
    ticket_id               TEXT NOT NULL,
    tool_name               TEXT NOT NULL,
    payload                 TEXT,
    risk_level              TEXT,
    requires_human_approval INTEGER,
    status                  TEXT NOT NULL,
    idempotency_key         TEXT NOT NULL UNIQUE,
    requested_by            TEXT,
    result                  TEXT,
    created_at              TEXT,
    executed_at             TEXT,
    FOREIGN KEY (ticket_id) REFERENCES tickets (ticket_id)
);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    action_id   TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    decision    TEXT NOT NULL,
    reason      TEXT,
    created_at  TEXT,
    FOREIGN KEY (action_id) REFERENCES tool_actions (action_id)
);

CREATE TABLE IF NOT EXISTS traces (
    run_id            TEXT PRIMARY KEY,
    ticket_id         TEXT,
    run_type          TEXT,
    status            TEXT,
    retrieved_doc_ids TEXT,
    tool_calls        TEXT,
    guardrail_results TEXT,
    escalation_reason TEXT,
    model_name        TEXT,
    prompt_version    TEXT,
    latency_ms        INTEGER,
    created_at        TEXT
);

CREATE TABLE IF NOT EXISTS eval_runs (
    eval_run_id  TEXT PRIMARY KEY,
    started_at   TEXT,
    completed_at TEXT,
    total_cases  INTEGER,
    model        TEXT,
    metrics      TEXT,
    case_results TEXT
);

CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders (customer_id);
CREATE INDEX IF NOT EXISTS idx_tickets_customer ON tickets (customer_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON doc_chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_actions_ticket ON tool_actions (ticket_id);
CREATE INDEX IF NOT EXISTS idx_traces_ticket ON traces (ticket_id);
"""

# Kept separate so the app still starts on a SQLite build without FTS5.
FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_fts USING fts5 (
    chunk_id UNINDEXED,
    doc_id   UNINDEXED,
    heading,
    text
);
"""

def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection with sane defaults for concurrent use."""
    conn = sqlite3.connect(
        str(db_path or DB_PATH),
        timeout=30.0,
        isolation_level=None,  # explicit transaction control
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def ensure_fts(conn: sqlite3.Connection) -> bool:
    """Create the full-text index if this SQLite build supports FTS5."""
    # executescript commits implicitly, so don't call this inside a transaction.
    try:
        conn.executescript(FTS_SCHEMA)
        return True
    except sqlite3.OperationalError:
        return False


def fts_available(conn: sqlite3.Connection) -> bool:
    """Whether this database has the full-text index."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'doc_chunks_fts'"
    ).fetchone()
    return row is not None


def init_db(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Create the schema if needed and return an open connection."""
    conn = connect(db_path)
    conn.executescript(SCHEMA)
    ensure_fts(conn)
    return conn


def reset_db(db_path: Path | str | None = None) -> None:
    """Drop the database file so a reload starts clean."""
    target = Path(db_path or DB_PATH)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(target) + suffix)
        if candidate.exists():
            candidate.unlink()


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def iter_rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> Iterator[sqlite3.Row]:
    cur = conn.execute(sql, params)
    try:
        yield from cur
    finally:
        cur.close()
