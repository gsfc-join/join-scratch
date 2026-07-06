"""IceChunk store schema: GCOM-W1 AMSR2 L3 Snow Depth.

This module implements the ``create_store`` interface required by
``handler.py``.  All store structure and metadata specific to the AMSR2
product lives here.

Interface
---------
    create_store(bucket, prefix) -> None

Adding a new data source means creating a new module in this directory that
exports the same ``create_store`` function.  The dispatcher in ``handler.py``
will import it automatically via the ``source`` field in the Lambda event.
"""

from __future__ import annotations

import logging
import os

import icechunk
import numpy as np
import pandas as pd
import zarr
from zarr.codecs import Zlib

log = logging.getLogger(__name__)

# ── store schema constants (must match sources/amsr2.py in populate_job) ───────
NLAT, NLON = 1800, 3600
CLAT = 72  # lat chunk size — matches JAXA HDF5 chunk layout

FILL_MISSING = np.int16(-32768)  # no valid data
FILL_NOOBS   = np.int16(-32767)  # no observation (clear sky / outside swath)
SCALE_FACTOR = 0.1               # multiply raw int16 by this to get cm

TIME_UNITS    = "days since 1970-01-01"
TIME_CALENDAR = "proleptic_gregorian"
TIME_EPOCH    = pd.Timestamp("1970-01-01")

PREALLOC_START = pd.Timestamp("2012-06-01")
PREALLOC_END   = pd.Timestamp("2030-12-31")

ORBIT_LABELS = ["Ascending", "Descending"]
BAND_LABELS  = ["snow_depth", "quality_flag"]

_LATS = np.linspace(89.95, -89.95, NLAT).astype("float32")
_LONS = np.linspace(0.05, 359.95, NLON).astype("float32")

_prealloc_days     = pd.date_range(PREALLOC_START, PREALLOC_END, freq="D")
_PREALLOC_DAY_VALS = ((_prealloc_days - TIME_EPOCH).days.astype("int32"))
_PREALLOC_NDAYS    = len(_PREALLOC_DAY_VALS)


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_repo_config() -> icechunk.RepositoryConfig:
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "https://gportal.jaxa.jp/", icechunk.http_store()
        )
    )
    return config


def _s3_storage(bucket: str, prefix: str) -> icechunk.Storage:
    region = os.environ.get("AWS_REGION", "us-west-2")
    return icechunk.s3_storage(bucket=bucket, prefix=prefix, region=region)


# ── public interface ───────────────────────────────────────────────────────────

def create_store(bucket: str, prefix: str) -> None:
    """Create a preallocated IceChunk store for AMSR2 L3 Snow Depth on S3."""
    log.info(
        "Creating AMSR2 IceChunk store at s3://%s/%s  (%d time slots, %s → %s)",
        bucket, prefix, _PREALLOC_NDAYS,
        PREALLOC_START.date(), PREALLOC_END.date(),
    )
    storage = _s3_storage(bucket, prefix)
    config  = _make_repo_config()
    repo    = icechunk.Repository.create(storage, config=config)

    session = repo.writable_session("main")
    root    = zarr.open_group(session.store, mode="w", zarr_format=3)

    root.require_array(
        "time",
        shape=(_PREALLOC_NDAYS,),
        chunks=(512,),
        dtype="int32",
        dimension_names=["time"],
        attributes={"units": TIME_UNITS, "calendar": TIME_CALENDAR, "long_name": "observation date"},
    )
    root["time"][:] = _PREALLOC_DAY_VALS

    root.require_array(
        "orbit",
        shape=(2,),
        chunks=(2,),
        dtype=str,
        dimension_names=["orbit"],
        attributes={"long_name": "orbit direction"},
    )
    root["orbit"][:] = ORBIT_LABELS

    root.require_array(
        "band",
        shape=(2,),
        chunks=(2,),
        dtype=str,
        dimension_names=["band"],
        attributes={"long_name": "band name"},
    )
    root["band"][:] = BAND_LABELS

    root.require_array(
        "lat",
        shape=(NLAT,),
        chunks=(NLAT,),
        dtype="float32",
        dimension_names=["lat"],
        attributes={"units": "degrees_north", "long_name": "latitude"},
    )
    root["lat"][:] = _LATS

    root.require_array(
        "lon",
        shape=(NLON,),
        chunks=(NLON,),
        dtype="float32",
        dimension_names=["lon"],
        attributes={"units": "degrees_east", "long_name": "longitude"},
    )
    root["lon"][:] = _LONS

    root.require_array(
        "geophysical_data",
        shape=(_PREALLOC_NDAYS, 2, NLAT, NLON, 2),
        chunks=(1, 1, CLAT, NLON, 2),
        dtype="int16",
        fill_value=int(FILL_MISSING),
        compressors=[Zlib(level=9)],
        dimension_names=["time", "orbit", "lat", "lon", "band"],
        attributes={
            "long_name":     "geophysical data",
            "units":         "cm",
            "scale_factor":  SCALE_FACTOR,
            "_FillValue":    int(FILL_MISSING),
            "missing_value": int(FILL_NOOBS),
        },
    )

    session.commit("Initialise preallocated store")
    log.info("AMSR2 IceChunk store created successfully.")
