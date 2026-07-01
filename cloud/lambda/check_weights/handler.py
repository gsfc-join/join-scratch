"""Lambda: check whether a weights file already exists on S3.

Used as the CheckWeightsCache state in the Step Functions state machine.
Returns a flag that a downstream Choice state uses to skip or run the
ESMF weight-generation Batch job.

Event schema (input)
--------------------
{
  "weights_uri": "s3://bucket/path/to/weights.nc4",
  ...                                               # all other keys passed through
}

Response schema (output)
-------------------------
All input keys forwarded unchanged, plus:
{
  "weights_exist": true | false
}
"""

from __future__ import annotations

import logging

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

_s3 = boto3.client("s3")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    without_scheme = uri[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


def handler(event: dict, context) -> dict:
    weights_uri = event.get("weights_uri", "").strip()

    if not weights_uri:
        raise ValueError("weights_uri is required in the event")
    if not weights_uri.startswith("s3://"):
        raise ValueError(f"weights_uri must be an s3:// URI, got: {weights_uri!r}")

    bucket, key = _parse_s3_uri(weights_uri)

    try:
        _s3.head_object(Bucket=bucket, Key=key)
        exists = True
        log.info("Weights cache HIT:  s3://%s/%s", bucket, key)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("404", "NoSuchKey"):
            exists = False
            log.info("Weights cache MISS: s3://%s/%s", bucket, key)
        else:
            # Unexpected error (permissions, etc.) — surface it.
            raise

    return {**event, "weights_exist": exists}
