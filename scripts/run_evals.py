"""Run the evaluation set and print a summary.

Usage:
    python -m scripts.run_evals                 # all cases, active model
    python -m scripts.run_evals --case eval_006 # a single case
    python -m scripts.run_evals --json          # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DB_PATH  # noqa: E402
from app.db import connect  # noqa: E402
from app.evals import format_report, run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TrustDesk evaluation cases.")
    parser.add_argument("--case", action="append", dest="cases", help="case id (repeatable)")
    parser.add_argument("--db", default=None, help="database path")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args()

    conn = connect(args.db or DB_PATH)
    try:
        report = run(conn, args.cases)
    finally:
        conn.close()

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))

    # Non-zero exit when any case failed, so CI can gate on it.
    return 0 if report["metrics"]["cases_fully_passed"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
