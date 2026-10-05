#!/usr/bin/env python

import os
# Silence icechunk rust warnings. Must be set before importing icechunk.
os.environ.setdefault("RUST_LOG", "error")

import argparse
import icechunk
from virtualizarr.parsers.hdf.hdf import _construct_manifest_array

import obstore
from obstore.store import HTTPStore

import io
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from zarr.codecs import Zlib
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Constants
NLAT, NLON = 1800, 3600          # HG grid dimensions
CLAT = 72                        # latitude chunk size (matches source HDF5)

# Updated fill values
FILL_MISSING = np.int16(-9999)   # fill value for missing retrievals
FILL_NOOBS = np.int16(-9999) 

SCALE_FACTOR = 0.1               # raw × 0.1 → cm

GPORTAL_BASE = (
    "https://gportal.jaxa.jp/download/standard/GCOM-W/GCOM-W.AMSR2"
    "/L3.SND_10/2/{yyyy}/{mm}/"
)
FNAME_TMPL = "GW1AM2_{date}_01D_{orbit}_L3SGSNDHG2210210.h5"
ORBIT_CODES = ["EQMA", "EQMD"]
ORBIT_LABELS = ["Ascending", "Descending"]

# CF time reference
TIME_UNITS = "days since 1970-01-01"
TIME_CALENDAR = "proleptic_gregorian"
TIME_EPOCH = pd.Timestamp("1970-01-01")

GPORTAL_URL = "https://gportal.jaxa.jp/"


def gportal_url(date: pd.Timestamp, orbit: str) -> str:
    """Return the G-Portal HTTPS URL for a date and orbit code."""
    return GPORTAL_BASE.format(
        yyyy=f"{date.year:04d}", mm=f"{date.month:02d}"
    ) + FNAME_TMPL.format(
        date=date.strftime("%Y%m%d"), orbit=orbit
    )


def _stream_hdf5(path: str, access_options: dict | None = None):
    """Return a context manager containing an HDF5 HTTP object in memory."""
    from urllib.parse import urlsplit

    if access_options is None:
        access_options = {}

    parts = urlsplit(path)
    base_url = f"{parts.scheme}://{parts.netloc}"
    obj_path = parts.path.lstrip("/")
    store = HTTPStore.from_url(base_url, **access_options)
    data = obstore.get(store, obj_path).bytes()
    buf = io.BytesIO(bytes(data))

    class _CM:
        def __enter__(self):
            return buf

        def __exit__(self, *_):
            buf.close()

    return _CM()


def _av(value):
    """Convert an HDF5 attribute value to a JSON-serialisable value."""
    if isinstance(value, np.ndarray):
        flat = value.flatten()
        if flat.dtype.kind in ("S", "U", "O"):
            items = [
                item.decode() if isinstance(item, bytes) else str(item)
                for item in flat
            ]
            return items[0] if len(items) == 1 else items
        values = flat.tolist()
        return values[0] if len(values) == 1 else values
    return value.decode() if isinstance(value, bytes) else value


def _get_manifests(chunk_url: str) -> tuple:
    """Download an HDF5 file and extract its VirtualiZarr manifest."""
    from urllib.parse import urlsplit

    parts = urlsplit(chunk_url)
    base_url = f"{parts.scheme}://{parts.netloc}"
    obj_path = parts.path.lstrip("/")
    store = HTTPStore.from_url(base_url)
    buf = io.BytesIO(obstore.get(store, obj_path).bytes())

    try:
        with h5py.File(buf, "r") as hdf:
            geophysical_data = hdf["Geophysical Data"]
            manifest_array = _construct_manifest_array(
                chunk_url, geophysical_data, "/"
            )
            attrs = {
                key: _av(value)
                for key, value in geophysical_data.attrs.items()
            }
        return manifest_array.manifest, attrs
    finally:
        buf.close()


def initialize_store(
    session: icechunk.Session,
    commit_msg: str = "Initialize empty store",
) -> None:
    """Create the arrays in a new IceChunk repository."""
    lats = np.linspace(89.95, -89.95, NLAT).astype("float32")
    lons = np.linspace(0.05, 359.95, NLON).astype("float32")

    root = zarr.open_group(session.store, mode="w", zarr_format=3)

    root.require_array(
        "time",
        shape=(0,),
        chunks=(512,),
        dtype="int32",
        dimension_names=["time"],
        attributes={
            "units": TIME_UNITS,
            "calendar": TIME_CALENDAR,
            "long_name": "observation date",
        },
    )

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
    root["band"][:] = ["snow_depth", "quality_flag"]

    root.require_array(
        "lat",
        shape=(NLAT,),
        chunks=(NLAT,),
        dtype="float32",
        dimension_names=["lat"],
        attributes={
            "units": "degrees_north",
            "long_name": "latitude",
        },
    )
    root["lat"][:] = lats

    root.require_array(
        "lon",
        shape=(NLON,),
        chunks=(NLON,),
        dtype="float32",
        dimension_names=["lon"],
        attributes={
            "units": "degrees_east",
            "long_name": "longitude",
        },
    )
    root["lon"][:] = lons

    root.require_array(
        "geophysical_data",
        shape=(0, 2, NLAT, NLON, 2),
        chunks=(1, 1, CLAT, NLON, 2),
        dtype="int16",
        fill_value=int(FILL_MISSING),
        compressors=[Zlib(level=9)],
        dimension_names=["time", "orbit", "lat", "lon", "band"],
        attributes={
            "long_name": (
                "geophysical data (band 0 = snow depth, "
                "band 1 = quality flag)"
            ),
            "units": "cm",
            "scale_factor": SCALE_FACTOR,
            "_FillValue": int(FILL_MISSING),
            "no_observation_value": int(FILL_NOOBS),
            "comment": (
                f"No observation value {int(FILL_NOOBS)} indicates ocean or "
                "areas without observation. This value is preserved in the "
                "raw data but is not treated as a fill value by CF conventions."
            ),
        },
    )

    session.commit(commit_msg)
    log.info("Initialized empty store")


def stored_days(repo: icechunk.Repository) -> list[int]:
    """Return stored day values in their current storage order."""
    session = repo.readonly_session("main")
    root = zarr.open_group(session.store, mode="r", zarr_format=3)
    return root["time"][:].tolist()


def insert_date(date: pd.Timestamp, repo: icechunk.Repository) -> None:
    """Insert a date using virtual references to the source HDF5 chunks."""
    day_val = int((date - TIME_EPOCH).days)
    date_iso = date.strftime("%Y-%m-%d")
    current = stored_days(repo)

    if day_val in current:
        slot = current.index(day_val)
        log.info("%s → skipped (already present, slot %d)", date_iso, slot)
        return

    orbit_manifests: dict[int, object] = {}
    for orbit_index, orbit in enumerate(ORBIT_CODES):
        chunk_url = gportal_url(date, orbit)
        try:
            manifest, _ = _get_manifests(chunk_url)
            orbit_manifests[orbit_index] = manifest
        except Exception as exc:
            log.warning(
                "  could not fetch %s for %s: %s",
                orbit,
                date_iso,
                exc,
            )

    if not orbit_manifests:
        log.info("%s → skipped (no orbits available on G-Portal)", date_iso)
        return

    new_slot = len(current)
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="r+", zarr_format=3)
    
    root["time"].resize((new_slot + 1,))
    root["time"][new_slot] = day_val
    root["geophysical_data"].resize((new_slot + 1, 2, NLAT, NLON, 2))

    # Each source chunk has shape (CLAT, NLON, 2), so both bands are
    # represented by the same virtual source chunk.
    for orbit_index, manifest in orbit_manifests.items():
        for (lat_chunk, lon_chunk, band_chunk), reference in manifest.iter_refs():
            key = (
                f"geophysical_data/c/{new_slot}/{orbit_index}/"
                f"{lat_chunk}/{lon_chunk}/{band_chunk}"
            )
            session.store.set_virtual_ref(
                key,
                reference["path"],
                offset=reference["offset"],
                length=reference["length"],
            )

    session.commit(f"Insert {date_iso} → slot {new_slot}")
    log.info("%s → inserted from on G-Portal, new slot = %d", date_iso, new_slot)

def main():
    parser = argparse.ArgumentParser(description="Create and populate AMSR2 icechunk store.")
    parser.add_argument(
        "--start-date", 
        type=str, 
        required=True, 
        help="Start date in YYYY-MM-DD format (e.g. 2018-09-01)"
    )
    parser.add_argument(
        "--end-date", 
        type=str, 
        required=True, 
        help="End date in YYYY-MM-DD format (e.g. 2019-07-01)"
    )
    args = parser.parse_args()

    bucket = "airborne-smce-prod-user-bucket"
    prefix = "JOIN/icechunk-stores/GCOM-W1-AMSR2-L3-SND"

    # ==========================================
    # Connect to Icechunk & Check Existing Inventory
    # ==========================================
    log.info(f"Connecting to Icechunk repository at s3://{bucket}/{prefix}...")
    storage = icechunk.s3_storage(bucket=bucket, prefix=prefix)

    config = icechunk.RepositoryConfig.default()
    container = icechunk.VirtualChunkContainer(
        name="amsr2_archive",       
        url_prefix=GPORTAL_URL,  
        store=icechunk.http_store()
    )
    config.set_virtual_chunk_container(container)

    repo = icechunk.Repository.open_or_create(
        storage=storage, 
        config=config,
        authorize_virtual_chunk_access={GPORTAL_URL: icechunk.credentials.HttpAccess}
    )

    session = repo.writable_session("main")

    # ==========================================
    # Check if this is the first run / empty store
    # ==========================================
    is_first_run = True
    try:
        root = zarr.open_group(session.store, mode="r", zarr_format=3)
        if "time" in root:
            is_first_run = False
            log.info("Found existing dataset! Arrays already initialized.")
    except Exception:
        pass

    if is_first_run:
        log.info("No existing dataset found (or store is empty). Running initial initialization...")
        initialize_store(session)

    # ==========================================
    # Fetch Data and Insert Dates
    # ==========================================
    date_seq = pd.date_range(
        start=args.start_date,
        end=args.end_date,
        freq="D",
    )

    log.info(f"Processing {len(date_seq)} dates from {args.start_date} to {args.end_date}...")
#    for date in tqdm(date_seq):
    for date in date_seq:
        insert_date(date, repo)


if __name__ == "__main__":
    main()