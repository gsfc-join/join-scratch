"""Lambda: apply ESMF weights and write regridded output — PLACEHOLDER.

This function is wired into the Step Functions state machine as the final
ApplyRegrid state but is not yet implemented.  It returns a structured
response indicating the inputs that are available for a future implementation.

Once implemented, this service should:
  1. Open the AMSR2 IceChunk store (read-only) at store_uri.
  2. Download the ESMF sparse weights from weights_uri.
  3. For each requested date, apply the NaN-safe sparse matrix multiply.
  4. Build a CF-compliant xarray.Dataset on the LIS LCC grid.
  5. Write to S3 as NetCDF and/or an IceChunk output store.

See workflows/swe/amsr2_regrid.py (load_weights, regrid_slice, build_dataset)
for the reference implementation.

Event schema (input)
--------------------
{
  "store_uri":   "s3://...",   # AMSR2 IceChunk store (read-only)
  "weights_uri": "s3://...",   # ESMF weights NetCDF on S3
  "lis_path":    "s3://...",   # LIS NetCDF for grid metadata
  "dates":       ["20230101", ...],
  ...
}
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


def handler(event: dict, context) -> dict:
    log.warning(
        "apply_regrid is not yet implemented. "
        "Consumers should open the IceChunk store and weights directly."
    )
    return {
        "status": "not_implemented",
        "message": (
            "Service 4 (apply ESMF weights + write regridded output) is a future TODO. "
            "Use the store_uri and weights_uri directly to apply weights yourself."
        ),
        "inputs_available": {
            "store_uri":   event.get("store_uri"),
            "weights_uri": event.get("weights_uri"),
            "lis_path":    event.get("lis_path"),
            "dates":       event.get("dates", []),
        },
    }
