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
from numcodecs import Shuffle

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Updated workflow from ASMR2 to GEOS-FP
# GEOS-FP is stored at e.g., https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y2024/M05/D06/
# netcdf file
# note: the storage changes at some point between 2019 and 2026
# currently takes about 5 minutes for a days worth of files (24 files)

# Constants
NLAT, NLON = 721, 1152           # GEOS grid dimensions
CLAT, CLON = 91, 144             # Latitude & longitude chunk sizes for 2019
# NOTE: the chunk sizes change between 2019 & 2026 --> may need to have seperate stores for each

# Fill values (from metadata)
FILL_MISSING  = 1.e+15 

#SCALE_FACTOR  = 1.0         

GEOS_BASE = (
    "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y{yyyy}/M{mm}/D{dd}/"
)

FNAME_TMPL   = "GEOS.fp.asm.tavg1_2d_slv_Nx.{date}_{time}.V01.nc4"
    
#STORE_URL = "s3://"
#STORE_DIR = Path("/tmp/amsr2-store-multi").resolve()

# CF time reference
TIME_UNITS    = "seconds since 1970-01-01"
TIME_CALENDAR = "proleptic_gregorian"
TIME_EPOCH    = pd.Timestamp("1970-01-01")

# helpers

def geos_url(date: pd.Timestamp) -> str:
    """Return the NCCS HTTPS URL for a given date (pd.Timestamp)."""
    yyyy = f"{date.year:04d}"
    mm = f"{date.month:02d}"
    dd = f"{date.day:02d}"
    
    # Format e.g., "20190604" and "1000"
    date_str = date.strftime("%Y%m%d")
    time_str = date.strftime("%H%M") 
    
    return GEOS_BASE.format(yyyy=yyyy, mm=mm, dd=dd) + FNAME_TMPL.format(
        date=date_str, time=time_str
    )

# def _get_manifests(chunk_url: str) -> object:
#     """Uses HTTP Range requests to extract the chunk manifest instantly.
#     Alternative to Alexey's code for this data?"""
    
#     # Open the dataset virtually over HTTPS
#     vds = open_virtual_dataset(chunk_url, indexes={}, loadable_variables=[])
    
#     # Extract the chunk manifest for T2M
#     manifest = vds["T2M"].data.manifest
#     return manifest

def _get_manifests(chunk_url: str) -> tuple:
    from urllib.parse import urlsplit
    
    # Break apart the URL
    parts    = urlsplit(chunk_url)
    base_url = f"{parts.scheme}://{parts.netloc}"
    obj_path = parts.path.lstrip("/")
    
    # 1. Download the NetCDF4 file into RAM using obstore
    store    = HTTPStore.from_url(base_url)
    buf      = io.BytesIO(obstore.get(store, obj_path).bytes())
    
    # 2. Read the headers with h5py and VirtualiZarr
    with h5py.File(buf, "r") as f:
        # Get the physical byte-locations for the T2M variable
        ma = _construct_manifest_array(chunk_url, f["T2M"], "/")
        
    # Return the manifest. We return an empty dictionary {} for the attributes 
    # since we are setting them manually in initialize_store.
    return ma.manifest, {}

def initialize_store(session: icechunk.Session, commit_msg: str = "Initialize empty store") -> None:
    """
    Create a fresh IceChunk repository at *store_dir*.

    Parameters
    ----------
    store_dir : Path
        Directory for the IceChunk repository.
    """

    lats = np.linspace(90.0, -90.0, NLAT).astype("float64") # match metadata
    lons = np.linspace(179.6875, -180.0, NLON).astype("float64") # match metadata

    root = zarr.open_group(session.store, mode="w", zarr_format=3)

    # ── coordinate arrays ─────────────────────────────────────────────────────
    # 1. TIME
    root.require_array(
        "time",
        shape=(0,),
        chunks=(512,),
        dtype="int64",
        dimension_names=["time"],
        attributes={
            "units": TIME_UNITS,
            "calendar": TIME_CALENDAR,
            "long_name": "observation date",
        },
    )
    # 2. LATITUDE
    root.require_array(
        "lat",
        shape=(NLAT,),
        chunks=(CLAT,),
        dtype="float64",
        dimension_names=["lat"],
        attributes={"units": "degrees_north", "long_name": "latitude"},
    )
    root["lat"][:] = lats
    
    # 3. lONGITUDE
    root.require_array(
        "lon",
        shape=(NLON,),
        chunks=(CLON,),
        dtype="float64",
        dimension_names=["lon"],
        attributes={"units": "degrees_east", "long_name": "longitude"},
    )
    root["lon"][:] = lons

    # ── data variable ─────────────────────────────────────────────────────────
    # Must be repeated for each variable. 
    # For this JOIN workflow, only pulling T2M.
    root.require_array(
        "T2M", # Temperature at 2 m above the displacement height (Units: K)
        shape=(0, NLAT, NLON),
        chunks=(1, CLAT, CLON),
        dtype="float32",
        fill_value=FILL_MISSING,
        compressors=[Zlib(level=2)],
        dimension_names=["time", "lat", "lon"],
        attributes={
            "long_name": "2-meter_air_temperature",
            "units": "K",
            #"scale_factor": SCALE_FACTOR
        },
    )

    session.commit(commit_msg)
    log.info("Initialized empty store")


def open_repo() -> icechunk.Repository:
    return icechunk.Repository.open(
        icechunk.local_filesystem_storage(str(STORE_DIR)),
        authorize_virtual_chunk_access={"https://portal.nccs.nasa.gov/": None},
    )


def stored_days(repo: icechunk.Repository) -> list[int]:
    """Return the list of int32 day values currently in the store (insertion order)."""
    session = repo.readonly_session("main")
    root    = zarr.open_group(session.store, mode="r", zarr_format=3)
    n = root["time"].shape[0]
    return list(root["time"][:].tolist()) if n > 0 else []


def insert_date(date: pd.Timestamp, repo: icechunk.Repository) -> None:
    """
    Insert *date* (YYYYMMDD) into the IceChunk store.

    Downloads each orbit's HDF5 file from G-Portal HTTPS into memory,
    extracts chunk byte offsets via VirtualiZarr, and writes virtual
    chunk references pointing back to the same HTTPS URLs.
    No data is copied; the store only records byte positions.
    Returns a human-readable status string.
    """
    time_val  = int((date - TIME_EPOCH).total_seconds())
    date_iso = date.strftime("%Y-%m-%d %H:%M")

    current = stored_days(repo)

    if time_val in current:
        slot = current.index(time_val)
        log.info(f"{date_iso}  →  skipped (already present, slot {slot})")
        return

     # Fetch the manifest for this specific date/time
    chunk_url = geos_url(date)
    try:
        manifest, _ = _get_manifests(chunk_url) # if using original function
        #manifest = _get_manifests(chunk_url) # if using the range request
    except Exception as exc:
        log.warning(f"  could not fetch data for {date_iso}: {exc}")
        return

    new_slot = len(current)

    session = repo.writable_session("main")
    store   = session.store
    root    = zarr.open_group(store, mode="r+", zarr_format=3)

    # Grow time dimension and T2M data array
    root["time"].resize((new_slot + 1,))
    root["time"][new_slot] = time_val
    root["T2M"].resize((new_slot + 1, NLAT, NLON))

    # Write virtual chunk refs.
    # The manifest key (ct, ci, cj) maps to (time, lat-chunk, lon-chunk)
    # ct is always zero because there is only 1 time slot per file
    for (ct, ci, cj), ref in manifest.iter_refs():
        key = f"T2M/c/{new_slot}/{ci}/{cj}"
        store.set_virtual_ref(
            key, ref["path"], offset=ref["offset"], length=ref["length"]
        )

    session.commit(f"Insert {date_iso} → slot {new_slot}")

################################################################################

# Initialize icechunk store
geosfp_url = "https://portal.nccs.nasa.gov/"

try:
    repo = icechunk.Repository.open(
        icechunk.s3_storage(
            bucket="airborne-smce-prod-user-bucket",
            prefix="JOIN/icechunk-stores/GEOS-FP-T2M"
        ),
        authorize_virtual_chunk_access={geosfp_url: None}
    )
    session = repo.writable_session("main")
    log.info("Found existing icechunk store")
except icechunk.IcechunkError as e:
    log.warning(
        f"Failed to open existing repo with error: {str(e)}. "
        "Trying to create a fresh repo."
    )
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(geosfp_url, icechunk.http_store())
    )
    repo = icechunk.Repository.create(
        icechunk.s3_storage(
            bucket="airborne-smce-prod-user-bucket",
            prefix="JOIN/icechunk-stores/GEOS-FP-T2M"
        ),
        config=config,
        authorize_virtual_chunk_access={geosfp_url: None}
    )
    session = repo.writable_session("main")
    initialize_store(session)

# time averaged data must start and end on half hour
date_seq = pd.date_range(start="2019-06-01 00:30", end="2019-06-01 23:30", freq="h") 

for date in tqdm(date_seq):
    insert_date(date, repo)

################################################################################

log.info("Testing reading a subset of data")
# Try reading the data
session = repo.readonly_session("main")
ds = xr.open_zarr(session.store)

sub = ds.sel(lat=52.07, lon=-91.99, method="nearest").sel(
    time=slice("2019-06-01", "2019-06-02"))
print(sub["T2M"].values)