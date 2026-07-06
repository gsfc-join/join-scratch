#!/usr/bin/env python
"""AWS Batch entrypoint: populate IceChunk virtual refs for one date.

Dispatches to the appropriate source module based on the ``SOURCE`` environment
variable.  All data-source-specific logic lives in ``sources/<name>.py``.

Adding a new data source
------------------------
1. Create ``sources/<name>.py`` that exports::

       def insert_date(bucket: str, prefix: str, date: str, force: bool) -> None: ...

2. Pass ``source=<name>`` in the Step Functions execution input.

Configuration — environment variables
--------------------------------------
Required:
  SOURCE            Data source identifier (e.g. ``amsr2``).
                    Must match a module name under ``sources/``.
  STORE_URI         s3:// URI for the IceChunk store root.
  DATE              Date in ``YYYYMMDD`` format.

Optional:
  FORCE_REPOPULATE  Set to "true" / "1" / "yes" to overwrite an already-
                    populated slot.  Default: false.
  AWS_REGION        AWS region (default: us-west-2).
  RUST_LOG          Rust log level (default: error).
"""

from __future__ import annotations

import importlib
import logging
import os
import sys

# Must be set BEFORE importing icechunk (Rust logger initialised at import time).
os.environ.setdefault("RUST_LOG", "error")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    without_scheme = uri[len("s3://"):]
    bucket, _, prefix = without_scheme.partition("/")
    return bucket, prefix


def main() -> None:
    source    = os.environ.get("SOURCE", "").strip()
    store_uri = os.environ.get("STORE_URI", "").strip()
    date      = os.environ.get("DATE", "").strip()

    if not source:
        log.error("Required environment variable SOURCE is not set.")
        sys.exit(1)
    if not store_uri:
        log.error("Required environment variable STORE_URI is not set.")
        sys.exit(1)
    if not date:
        log.error("Required environment variable DATE is not set (YYYYMMDD).")
        sys.exit(1)
    if len(date) != 8 or not date.isdigit():
        log.error("DATE must be in YYYYMMDD format, got: %r", date)
        sys.exit(1)

    force = os.environ.get("FORCE_REPOPULATE", "").lower() in ("1", "true", "yes")

    log.info("SOURCE           : %s", source)
    log.info("STORE_URI        : %s", store_uri)
    log.info("DATE             : %s", date)
    log.info("FORCE_REPOPULATE : %s", force)

    # ── load source module ─────────────────────────────────────────────────────
    try:
        mod = importlib.import_module(f"sources.{source}")
    except ModuleNotFoundError:
        log.error(
            "Unknown source %r — no module sources/%s.py found. "
            "Available sources: check the sources/ directory.",
            source, source,
        )
        sys.exit(1)

    if not hasattr(mod, "insert_date"):
        log.error("sources/%s.py does not export insert_date().", source)
        sys.exit(1)

    bucket, prefix = _parse_s3_uri(store_uri)
    mod.insert_date(bucket, prefix, date, force=force)
    log.info("Done.")


if __name__ == "__main__":
    main()
