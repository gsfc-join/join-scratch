"""IceChunk populate source: GCOM-W1 AMSR2 L3 Snow Depth (G-Portal).

This module implements the ``insert_date`` interface required by
``populate_job.py``.  All logic specific to JAXA G-Portal, AMSR2 HDF5
structure, orbit codes, and the store schema lives here.

Interface
---------
    insert_date(bucket, prefix, date, force) -> None

Adding a new data source means creating a new module in this directory that
exports the same ``insert_date`` function.  The dispatcher in
``populate_job.py`` will import it automatically via the ``SOURCE`` env var.
"""

from __future__ import annotations

import io
import logging
import sys
from urllib.parse import urlsplit

import h5py
import icechunk
import numpy as np
import obstore
import pandas as pd
from obstore.store import HTTPStore
from virtualizarr.parsers.hdf.hdf import _construct_manifest_array

log = logging.getLogger(__name__)

# ── store schema constants (must match schemas/amsr2.py in init_store Lambda) ──
TIME_EPOCH     = pd.Timestamp("1970-01-01")
PREALLOC_START = pd.Timestamp("2012-06-01")
PREALLOC_END   = pd.Timestamp("2030-12-31")

_prealloc_days     = pd.date_range(PREALLOC_START, PREALLOC_END, freq="D")
_PREALLOC_DAY_VALS = (_prealloc_days - TIME_EPOCH).days.astype("int32")
_DAY_TO_SLOT: dict[int, int] = {int(d): i for i, d in enumerate(_PREALLOC_DAY_VALS)}

# ── Source grid spec (used by regrid_job.py to build the SCRIP source grid) ────
# AMSR2 L3 is always a fixed global 0.1° equirectangular grid.

# ── AMSR2 / G-Portal constants ──────────────────────────────────────────────────
GPORTAL_BASE = (
    "https://gportal.jaxa.jp/download/standard/GCOM-W/GCOM-W.AMSR2"
    "/L3.SND_10/2/{yyyy}/{mm}/"
)
FNAME_TMPL  = "GW1AM2_{date}_01D_{orbit}_L3SGSNDHG2210210.h5"
ORBIT_CODES = ["EQMA", "EQMD"]  # Ascending, Descending


# ── helpers ────────────────────────────────────────────────────────────────────

def _gportal_url(date: str, orbit: str) -> str:
    """Build the JAXA G-Portal HTTPS URL for one AMSR2 HDF5 file."""
    yyyy, mm = date[:4], date[4:6]
    return GPORTAL_BASE.format(yyyy=yyyy, mm=mm) + FNAME_TMPL.format(
        date=date, orbit=orbit
    )


def _date_to_days(date: str) -> int:
    ts = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:]}")
    return int((ts - TIME_EPOCH).days)


def _av(v):
    """Normalise an HDF5 attribute value to a plain Python scalar or list."""
    if isinstance(v, np.ndarray):
        flat = v.flatten()
        if flat.dtype.kind in ("S", "U", "O"):
            items = [x.decode() if isinstance(x, bytes) else str(x) for x in flat]
            return items[0] if len(items) == 1 else items
        lst = flat.tolist()
        return lst[0] if len(lst) == 1 else lst
    return v.decode() if isinstance(v, bytes) else v


def _make_repo_config() -> icechunk.RepositoryConfig:
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "https://gportal.jaxa.jp/", icechunk.http_store()
        )
    )
    return config


def _s3_storage(bucket: str, prefix: str) -> icechunk.Storage:
    import os
    region = os.environ.get("AWS_REGION", "us-west-2")
    return icechunk.s3_storage(bucket=bucket, prefix=prefix, region=region)


def _get_manifests(chunk_url: str):
    """Download one G-Portal HDF5 file and extract its VirtualiZarr manifest."""
    parts    = urlsplit(chunk_url)
    base_url = f"{parts.scheme}://{parts.netloc}"
    obj_path = parts.path.lstrip("/")
    store    = HTTPStore.from_url(base_url)
    buf      = io.BytesIO(obstore.get(store, obj_path).bytes())
    with h5py.File(buf, "r") as f:
        ma    = _construct_manifest_array(chunk_url, f["Geophysical Data"], "/")
        attrs = {k: _av(v) for k, v in f["Geophysical Data"].attrs.items()}
    return ma.manifest, attrs


def _slot_has_virtual_refs(session_store, slot: int) -> bool:
    test_key = f"geophysical_data/c/{slot}/0/0/0/0"
    try:
        ref = session_store.get_virtual_ref(test_key)
        return ref is not None
    except Exception:
        return False


# ── public interface ───────────────────────────────────────────────────────────

def insert_date(bucket: str, prefix: str, date: str, force: bool = False) -> None:
    """Insert virtual refs for one AMSR2 date into the IceChunk store.

    Opens a fresh repo session every call to always start from the latest
    committed snapshot and avoid ``ConflictError`` from stale parent refs.
    """
    day_val = _date_to_days(date)
    if day_val not in _DAY_TO_SLOT:
        log.warning(
            "Date %s (day=%d) is outside the preallocated range %s→%s — skipping.",
            date, day_val, PREALLOC_START.date(), PREALLOC_END.date(),
        )
        return
    slot = _DAY_TO_SLOT[day_val]

    repo    = icechunk.Repository.open(
        _s3_storage(bucket, prefix),
        config=_make_repo_config(),
        authorize_virtual_chunk_access={"https://gportal.jaxa.jp/": None},
    )
    session = repo.writable_session("main")
    store   = session.store

    if not force and _slot_has_virtual_refs(store, slot):
        log.info("Slot %d (%s) already populated — skipping.", slot, date)
        return

    log.info("Fetching G-Portal manifests for %s → slot %d …", date, slot)
    any_written = False
    for o_idx, orbit in enumerate(ORBIT_CODES):
        url = _gportal_url(date, orbit)
        try:
            manifest, _ = _get_manifests(url)
        except Exception as exc:
            log.warning(
                "Could not fetch manifest for %s orbit %s: %s", date, orbit, exc
            )
            continue
        for (ci, cj, ck), ref in manifest.iter_refs():
            key = f"geophysical_data/c/{slot}/{o_idx}/{ci}/{cj}/{ck}"
            store.set_virtual_ref(
                key, ref["path"], offset=ref["offset"], length=ref["length"]
            )
        any_written = True
        log.debug("  Wrote virtual refs for orbit %s (slot %d)", orbit, slot)

    if any_written:
        session.commit(f"Fill slot {slot} ({date})")
        log.info("Committed slot %d (%s).", slot, date)
    else:
        log.warning("No manifests written for %s — nothing committed.", date)
        sys.exit(1)
