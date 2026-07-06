"""Lambda: initialise a preallocated IceChunk store on S3.

Dispatches to the appropriate schema module based on the ``source`` field in
the Lambda event.  All store structure and metadata specific to a data source
lives in ``schemas/<source>.py``.

Adding a new data source
------------------------
1. Create ``schemas/<name>.py`` that exports::

       def create_store(bucket: str, prefix: str) -> None: ...

2. Pass ``"source": "<name>"`` in the Step Functions execution input.

Event schema (input)
--------------------
{
  "source":         "amsr2",        # required — selects schemas/<source>.py
  "store_uri":      "s3://bucket/path/to/store",  # required
  "recreate_store": false,          # optional bool or string "true"/"false"
  ...                               # all other keys passed through unchanged
}

Response schema (output)
------------------------
All input keys forwarded unchanged, plus:
{
  "store_initialized": true
}
"""

from __future__ import annotations

import importlib
import logging
import os

os.environ.setdefault("RUST_LOG", "error")

import obstore

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    without_scheme = uri[len("s3://"):]
    bucket, _, prefix = without_scheme.partition("/")
    return bucket, prefix


def _store_exists(bucket: str, prefix: str) -> bool:
    store = obstore.store.S3Store.from_url(
        f"s3://{bucket}/{prefix}",
        region=os.environ.get("AWS_REGION", "us-west-2"),
    )
    try:
        pages = list(obstore.list(store))
        keys = [obj["path"] for page in pages for obj in page]
        return len(keys) > 0
    except Exception:
        return False


def _delete_store(bucket: str, prefix: str) -> int:
    store = obstore.store.S3Store.from_url(
        f"s3://{bucket}/{prefix}",
        region=os.environ.get("AWS_REGION", "us-west-2"),
    )
    pages = list(obstore.list(store))
    keys = [obj["path"] for page in pages for obj in page]
    if keys:
        obstore.delete(store, keys)
    return len(keys)


def handler(event: dict, context) -> dict:
    source = event.get("source", "").strip()
    if not source:
        raise ValueError("'source' is required in the event (e.g. \"amsr2\")")

    store_uri = event.get("store_uri", "").strip()
    if not store_uri:
        raise ValueError("'store_uri' is required in the event")
    if not store_uri.startswith("s3://"):
        raise ValueError(f"store_uri must be an s3:// URI, got: {store_uri!r}")

    recreate_raw = event.get("recreate_store", False)
    if isinstance(recreate_raw, str):
        recreate = recreate_raw.lower() in ("1", "true", "yes")
    else:
        recreate = bool(recreate_raw)

    # ── load schema module ─────────────────────────────────────────────────────
    try:
        mod = importlib.import_module(f"schemas.{source}")
    except ModuleNotFoundError:
        raise ValueError(
            f"Unknown source {source!r} — no module schemas/{source}.py found. "
            "Add a schemas/<source>.py that exports create_store(bucket, prefix)."
        )

    if not hasattr(mod, "create_store"):
        raise ValueError(f"schemas/{source}.py does not export create_store().")

    bucket, prefix = _parse_s3_uri(store_uri)

    if _store_exists(bucket, prefix):
        if recreate:
            log.info("recreate_store=true — deleting existing store objects …")
            n = _delete_store(bucket, prefix)
            log.info("Deleted %d objects from s3://%s/%s", n, bucket, prefix)
        else:
            log.info(
                "Store already exists at s3://%s/%s — nothing to do.",
                bucket, prefix,
            )
            return {**event, "store_initialized": True}

    mod.create_store(bucket, prefix)
    return {**event, "store_initialized": True}
