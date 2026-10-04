"""Load the provided seed data into SQLite.

Usage:
    python -m scripts.seed            # load into trustdesk.db
    python -m scripts.seed --reset    # drop the database first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DB_PATH  # noqa: E402
from app.db import reset_db  # noqa: E402
from app.loader import load_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the TrustDesk database.")
    parser.add_argument("--reset", action="store_true", help="delete the database first")
    parser.add_argument("--db", default=None, help="database path (default trustdesk.db)")
    args = parser.parse_args()

    target = args.db or DB_PATH
    if args.reset:
        reset_db(target)
        print(f"reset  {target}")

    summary = load_all(target)
    print(f"loaded into {target}")
    for key in ("customers", "orders", "tickets", "tools", "eval_cases",
                "kb_documents", "kb_chunks"):
        print(f"  {key:<14} {summary[key]}")
    print(f"  {'embedder':<14} {summary['embedder']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
