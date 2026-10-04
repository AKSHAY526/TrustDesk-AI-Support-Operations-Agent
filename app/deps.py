"""Shared FastAPI dependencies."""

from __future__ import annotations

import sqlite3
from typing import Iterator

from .db import init_db


def get_db() -> Iterator[sqlite3.Connection]:
    """One connection per request, always closed."""
    conn = init_db()
    try:
        yield conn
    finally:
        conn.close()
