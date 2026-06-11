#!/usr/bin/env python

import os
# Silence icechunk rust warnings. Must be set before importing icechunk.
os.environ.setdefault("RUST_LOG", "error")

import icechunk
from virtualizarr.parsers.hdf.hdf import _construct_manifest_array

import obstore
from obstore.store import HTTPStore

import shutil
import io
from pathlib import Path
import logging

import numpy as np
import h5py
import zarr
from zarr.codecs import Zlib
import xarray as xr
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# constants
NLAT, NLON = 1800, 3600          # HG grid dimensions
CLAT       = 72                  # latitude chunk size (matches source HDF5)

# Fill values (raw int16, before scale factor)
FILL_MISSING  = np.int16(-32768)   # cloudy / no retrieval
FILL_NOOBS    = np.int16(-32767)   # ocean / no observation

SCALE_FACTOR  = 0.1               # raw × 0.1 → cm

GPORTAL_BASE = (
    "https://gportal.jaxa.jp/download/standard/GCOM-W/GCOM-W.AMSR2"
    "/L3.SND_10/2/{yyyy}/{mm}/"
)
FNAME_TMPL   = "GW1AM2_{date}_01D_{orbit}_L3SGSNDHG2210210.h5"
ORBIT_CODES  = ["EQMA",        "EQMD"       ]
ORBIT_LABELS = ["Ascending",   "Descending" ]

STORE_DIR = Path("/tmp/amsr2-store-multi").resolve()

# CF time reference
TIME_UNITS    = "days since 1970-01-01"
TIME_CALENDAR = "proleptic_gregorian"
TIME_EPOCH    = pd.Timestamp("1970-01-01")

# helpers

def gportal_url(date: pd.Timestamp, orbit: str) -> str:
    """Return the G-Portal HTTPS URL for a given date (pd.Timestamp) and orbit code."""
    yyyy = f"{date.year:04d}"
    mm = f"{date.month:02d}"
    return GPORTAL_BASE.format(yyyy=yyyy, mm=mm) + FNAME_TMPL.format(
        date=date.strftime("%Y%m%d"), orbit=orbit
    )


def _stream_hdf5(path: str, access_options: dict = {}):
    from urllib.parse import urlsplit

    parts = urlsplit(path)
    base_url = f"{parts.scheme}://{parts.netloc}"
    obj_path = parts.path.lstrip("/")
    store = HTTPStore.from_url(base_url, **access_options)
    data = obstore.get(store, obj_path).bytes()
    buf = io.BytesIO(bytes(data))

    class _CM:
        def __enter__(_):
            return buf

        def __exit__(_, *__):
            buf.close()

    return _CM()


def _av(v):
    """Convert an HDF5 attribute value to a JSON-serialisable Python scalar."""
    if isinstance(v, np.ndarray):
        flat = v.flatten()
        if flat.dtype.kind in ("S", "U", "O"):
            items = [x.decode() if isinstance(x, bytes) else str(x) for x in flat]
            return items[0] if len(items) == 1 else items
        lst = flat.tolist()
        return lst[0] if len(lst) == 1 else lst
    return v.decode() if isinstance(v, bytes) else v

def _get_manifests(chunk_url: str) -> tuple:
    """
    Stream the HDF5 file at *chunk_url* into memory and extract the
    VirtualiZarr manifest for the ``Geophysical Data`` group.

    Uses ``obstore`` to download the entire file in one HTTP request
    (``obstore_stream`` strategy), which is much faster than lazy fsspec
    streaming for HDF5 files that are not cloud-optimised.

    Returns (manifest, attrs) for the ``Geophysical Data`` group.
    Band 0 = snow depth, band 1 = quality flag.  The source array has shape
    (lat, lon, band) with chunks (CLAT, NLON, 2), so each chunk covers both
    bands.
    """
    from urllib.parse import urlsplit
    parts    = urlsplit(chunk_url)
    base_url = f"{parts.scheme}://{parts.netloc}"
    obj_path = parts.path.lstrip("/")
    store    = HTTPStore.from_url(base_url)
    buf      = io.BytesIO(obstore.get(store, obj_path).bytes())
    with h5py.File(buf, "r") as f:
        ma    = _construct_manifest_array(chunk_url, f["Geophysical Data"], "/")
        attrs = {k: _av(v) for k, v in f["Geophysical Data"].attrs.items()}
    return ma.manifest, attrs


def create_store(store_dir: Path, overwrite: bool = False) -> icechunk.Repository:
    """
    Create a fresh IceChunk repository at *store_dir*.

    Parameters
    ----------
    store_dir : Path
        Directory for the IceChunk repository.
    overwrite : bool
        If True, delete any existing repository first.
    """
    if overwrite:
        shutil.rmtree(store_dir, ignore_errors=True)
    store_dir.mkdir(parents=True, exist_ok=True)

    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "https://gportal.jaxa.jp/", icechunk.http_store()
        )
    )
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(store_dir)), config=config
    )

    lats = np.linspace(89.95, -89.95, NLAT).astype("float32")
    lons = np.linspace(0.05,  359.95, NLON).astype("float32")

    session = repo.writable_session("main")
    root    = zarr.open_group(session.store, mode="w", zarr_format=3)

    # ── coordinate arrays ─────────────────────────────────────────────────────
    root.require_array(
        "time",
        shape=(0,), chunks=(512,), dtype="int32",
        dimension_names=["time"],
        attributes={
            "units":    TIME_UNITS,
            "calendar": TIME_CALENDAR,
            "long_name": "observation date",
        },
    )
    root.require_array(
        "orbit",
        shape=(2,), chunks=(2,), dtype=str,
        dimension_names=["orbit"],
        attributes={"long_name": "orbit direction"},
    )
    root["orbit"][:] = ORBIT_LABELS

    root.require_array(
        "band",
        shape=(2,), chunks=(2,), dtype=str,
        dimension_names=["band"],
        attributes={"long_name": "band name"},
    )
    root["band"][:] = ["snow_depth", "quality_flag"]

    root.require_array(
        "lat",
        shape=(NLAT,), chunks=(NLAT,), dtype="float32",
        dimension_names=["lat"],
        attributes={"units": "degrees_north", "long_name": "latitude"},
    )
    root["lat"][:] = lats

    root.require_array(
        "lon",
        shape=(NLON,), chunks=(NLON,), dtype="float32",
        dimension_names=["lon"],
        attributes={"units": "degrees_east", "long_name": "longitude"},
    )
    root["lon"][:] = lons

    # ── data variable ─────────────────────────────────────────────────────────
    # Zlib(level=9) matches the zlib/deflate compression in the source HDF5 files.
    # The band dimension (size 2) is interleaved within each source chunk, so
    # we keep it as the innermost dimension to match the physical layout.
    root.require_array(
        "geophysical_data",
        shape=(0, 2, NLAT, NLON, 2),
        chunks=(1, 1, CLAT, NLON, 2),
        dtype="int16",
        fill_value=int(FILL_MISSING),
        compressors=[Zlib(level=9)],
        dimension_names=["time", "orbit", "lat", "lon", "band"],
        attributes={
            "long_name":    "geophysical data (band 0 = snow depth, band 1 = quality flag)",
            "units":        "cm",
            "scale_factor": SCALE_FACTOR,
            "_FillValue":   int(FILL_MISSING),
            "missing_value": int(FILL_NOOBS),
        },
    )

    session.commit("Initialise empty store")
    log.info(f"Created store at {store_dir}")
    return repo


def open_repo() -> icechunk.Repository:
    return icechunk.Repository.open(
        icechunk.local_filesystem_storage(str(STORE_DIR)),
        authorize_virtual_chunk_access={"https://gportal.jaxa.jp/": None},
    )


def stored_days(repo: icechunk.Repository) -> list[int]:
    """Return the list of int32 day values currently in the store (insertion order)."""
    session = repo.readonly_session("main")
    root    = zarr.open_group(session.store, mode="r", zarr_format=3)
    n = root["time"].shape[0]
    return list(root["time"][:].tolist()) if n > 0 else []


def insert_date(date: pd.Timestamp) -> None:
    """
    Insert *date* (YYYYMMDD) into the IceChunk store.

    Downloads each orbit's HDF5 file from G-Portal HTTPS into memory,
    extracts chunk byte offsets via VirtualiZarr, and writes virtual
    chunk references pointing back to the same HTTPS URLs.
    No data is copied; the store only records byte positions.
    Returns a human-readable status string.
    """
    day_val  = int((date - TIME_EPOCH).days)
    date_iso = date.strftime("%Y-%m-%d")

    repo    = open_repo()
    current = stored_days(repo)

    if day_val in current:
        slot = current.index(day_val)
        log.info(f"{date_iso}  →  skipped (already present, slot {slot})")
        return

    # Build manifests for each orbit by streaming the HDF5 from G-Portal.
    orbit_manifests: dict[int, object] = {}
    for o_idx, orbit in enumerate(ORBIT_CODES):
        chunk_url = gportal_url(date, orbit)
        try:
            manifest, _ = _get_manifests(chunk_url)
            orbit_manifests[o_idx] = manifest
        except Exception as exc:
            log.warning(f"  could not fetch {orbit} for {date_iso}: {exc}")

    if not orbit_manifests:
        log.info(f"{date_iso}  →  skipped (no orbits available on G-Portal)")
        return

    new_slot = len(current)

    session = repo.writable_session("main")
    store   = session.store
    root    = zarr.open_group(store, mode="r+", zarr_format=3)

    # Grow time dimension and data array
    root["time"].resize((new_slot + 1,))
    root["time"][new_slot] = day_val
    root["geophysical_data"].resize((new_slot + 1, 2, NLAT, NLON, 2))

    # Write virtual chunk refs.
    # Source chunks have shape (CLAT, NLON, 2) — both bands in one chunk.
    # The manifest key (ci, cj, ck) maps to (lat-chunk, lon-chunk, band-chunk).
    # Since chunk size covers the full band dim, ck is always 0.
    for o_idx, manifest in orbit_manifests.items():
        for (ci, cj, ck), ref in manifest.iter_refs():
            key = f"geophysical_data/c/{new_slot}/{o_idx}/{ci}/{cj}/{ck}"
            store.set_virtual_ref(
                key, ref["path"], offset=ref["offset"], length=ref["length"]
            )

    session.commit(f"Insert {date_iso} → slot {new_slot}")

################################################################################


repo = create_store(STORE_DIR, overwrite=True)

date_seq = pd.date_range(start="2018-09-01", end="2019-07-01", freq="D")

for date in date_seq:
    insert_date(date)

################################################################################

fname = "GW1AM2_20260507_01D_EQMA_L3SGSNDHG2210210.h5"
https_url = (
    "https://gportal.jaxa.jp/download/standard/GCOM-W/GCOM-W.AMSR2"
    f"/L3.SND_10/2/2026/05/{fname}"
)

local_store = Path("/tmp") / "amsr2-store-01"

make_icechunk_repo(
    str(local_store),
    https_url,
    None,
    vc_prefix=f"https://gportal.jaxa.jp/",
    vc_store_config=icechunk.http_store()
)

store = icechunk.local_filesystem_storage(local_store)
repo = icechunk.Repository.open(store, authorize_virtual_chunk_access={"https://gportal.jaxa.jp/": None})
session = repo.readonly_session("main")
ds = xr.open_zarr(session.store, consolidated=False, zarr_format=3)

ak = ds["Geophysical_Data"].isel(lat=slice(180, 360), lon=slice(1880, 2300))
ak.mean().values
