"""Lambda: enumerate dates between start_date and end_date (inclusive).

Invoked as the first state in the Step Functions state machine.  Builds the
list of YYYYMMDD date strings that the downstream Map state fans out across
Batch populate jobs.

Event schema (input)
--------------------
{
  "start_date":    "20230101",   # required, YYYYMMDD
  "end_date":      "20230110",   # optional, defaults to start_date
  "store_uri":     "s3://...",   # passed through unchanged
  "weights_uri":   "s3://...",   # passed through unchanged
  "lis_path":      "s3://...",   # passed through unchanged
  "method":        "bilinear",   # passed through unchanged
  "recreate_store":    false,    # passed through unchanged
  "force_repopulate":  false,    # passed through unchanged
  "force_regenerate":  false     # passed through unchanged
}

Response schema (output)
-------------------------
All input keys are forwarded unchanged, plus:
{
  "dates": ["20230101", "20230102", ..., "20230110"]
}
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _parse_date(s: str) -> date:
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"Expected YYYYMMDD, got: {s!r}")
    return date(int(s[:4]), int(s[4:6]), int(s[6:]))


def handler(event: dict, context) -> dict:
    """Enumerate dates and inject them into the Step Functions execution state."""
    start_str = event.get("start_date", "")
    end_str   = event.get("end_date", start_str)

    if not start_str:
        raise ValueError("start_date is required")

    start = _parse_date(start_str)
    end   = _parse_date(end_str)

    if end < start:
        raise ValueError(
            f"end_date {end_str} must be >= start_date {start_str}"
        )

    dates: list[str] = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    log.info(
        "Enumerated %d dates: %s → %s",
        len(dates), dates[0], dates[-1],
    )

    return {**event, "dates": dates}
