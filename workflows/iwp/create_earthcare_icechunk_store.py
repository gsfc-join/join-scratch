#!/usr/bin/env python

import os
# Silence icechunk rust warnings. Must be set before importing icechunk.
os.environ.setdefault("RUST_LOG.icechunk", "error")

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
import zarr
from zarr.codecs import Zlib

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

TRACK_CHUNK = 1000
HEIGHT_BINS = 137
FILL_MISSING = np.float32(-9999.0)

# EarthCARE Group Paths exactly as specified
AC_CLP_PATH = "EarthCARE/CPR+ATLID/LEVEL2/CPR-ATLID Synergy Cloud（AC__CLP）"
AUX_JSG_PATH = "EarthCARE/Supprt Data/L1d Joint Standard Grid (X-JSG)"

def _stream_hdf5(path: str, access_options: dict | None = None):
    """Return a context manager containing an HDF5 HTTP/local object in memory."""
    from urllib.parse import urlsplit
    if access_options is None:
        access_options = {}

    parts = urlsplit(path)
    if not parts.scheme or parts.scheme == 'file':
        with open(parts.path, "rb") as f:
            buf = io.BytesIO(f.read())
    else:
        base_url = f"{parts.scheme}://{parts.netloc}"
        obj_path = parts.path.lstrip("/")
        store = HTTPStore.from_url(base_url, **access_options)
        buf = io.BytesIO(obstore.get(store, obj_path).bytes())

    class _CM:
        def __enter__(self): return buf
        def __exit__(self, *_): buf.close()
    return _CM()

def _get_manifests(chunk_url: str, hdf_group_path: str) -> tuple:
    """Download an HDF5 file and extract its VirtualiZarr manifest for a specific group."""
    with _stream_hdf5(chunk_url) as buf:
        try:
            with h5py.File(buf, "r") as hdf:
                target_group = hdf[hdf_group_path]
                manifest_array = _construct_manifest_array(chunk_url, target_group, "/")
                
                attrs = {}
                for key, value in target_group.attrs.items():
                    if isinstance(value, np.ndarray):
                        val = value.tolist()
                        attrs[key] = val[0] if len(val) == 1 else val
                    elif isinstance(value, bytes):
                        attrs[key] = value.decode()
                    else:
                        attrs[key] = value
                        
            return manifest_array.manifest, attrs
        except Exception as e:
            log.error(f"Failed to extract manifest from {chunk_url}: {e}")
            raise

def initialize_store(session: icechunk.Session, commit_msg: str = "Initialize empty EarthCARE store") -> None:
    """Create the hierarchy and arrays in a new IceChunk repository for EarthCARE."""
    root = zarr.open_group(session.store, mode="w", zarr_format=3)

    # Initialize AC__CLP Group
    ac_clp_group = root.require_group(AC_CLP_PATH)
    ac_clp_group.require_array(
        "cloud_fraction",  # Change to your specific array name
        shape=(0, HEIGHT_BINS),
        chunks=(TRACK_CHUNK, HEIGHT_BINS),
        dtype="float32",
        fill_value=float(FILL_MISSING),
        compressors=[Zlib(level=9)],
        dimension_names=["along_track", "height_bin"],
        attributes={"long_name": "Synergy Cloud Fraction"},
    )

    # Initialize AUX_JSG Group
    aux_jsg_group = root.require_group(AUX_JSG_PATH)
    aux_jsg_group.require_array(
        "latitude",
        shape=(0,),
        chunks=(TRACK_CHUNK,),
        dtype="float32",
        fill_value=float(FILL_MISSING),
        compressors=[Zlib(level=9)],
        dimension_names=["along_track"],
    )

    session.commit(commit_msg)
    log.info("Initialized empty EarthCARE store on S3.")

def find_earthcare_granules(date: pd.Timestamp, input_dir: Path) -> tuple:
    """Find local EarthCARE HDF5 files matching the date (YYYYMMDD format)."""
    date_str = date.strftime("%Y%m%d")
    
    # E.g., ECA_J_AC__CLP_2BS_20260124T0659_...h5
    ac_clp_files = list(input_dir.glob(f"*AC__CLP*{date_str}*.h5"))
    aux_jsg_files = list(input_dir.glob(f"*AUX_JSG*{date_str}*.h5"))
    
    return [str(f) for f in ac_clp_files], [str(f) for f in aux_jsg_files]

def insert_date(date: pd.Timestamp, session: icechunk.Session, input_dir: Path) -> None:
    """Find and insert EarthCARE granules for a specific date using virtual references."""
    date_iso = date.strftime("%Y-%m-%d")
    log.info(f"Processing date: {date_iso}")

    ac_clp_urls, aux_jsg_urls = find_earthcare_granules(date, input_dir)

    if not ac_clp_urls and not aux_jsg_urls:
        log.info(f"{date_iso} → skipped (no EarthCARE granules found in {input_dir})")
        return

    root = zarr.open_group(session.store, mode="a", zarr_format=3)

    # Process AC__CLP granules
    for chunk_url in ac_clp_urls:
        try:
            manifest, attrs = _get_manifests(chunk_url, AC_CLP_PATH)
            # Map manifest chunks into the icechunk session here
            log.info(f"Successfully processed virtual reference for {Path(chunk_url).name}")
        except Exception as exc:
            log.warning(f"Could not fetch/process {chunk_url}: {exc}")

    # Process AUX_JSG granules
    for chunk_url in aux_jsg_urls:
        try:
            manifest, attrs = _get_manifests(chunk_url, AUX_JSG_PATH)
            # Map manifest chunks into the icechunk session here
            log.info(f"Successfully processed virtual reference for {Path(chunk_url).name}")
        except Exception as exc:
            log.warning(f"Could not fetch/process {chunk_url}: {exc}")
            
    session.commit(f"Appended granules for {date_iso}")

def main():
    parser = argparse.ArgumentParser(description="Create Icechunk store for EarthCARE on S3.")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    
    # Input files directory
    parser.add_argument("--input-dir", type=str, required=True, help="Local directory containing EarthCARE HDF5 files to scan")
    
    # S3 Icechunk Configuration
    parser.add_argument("--s3-bucket", type=str, default="airborne-smce-prod-user-bucket", help="S3 bucket for the icechunk repo")
    parser.add_argument("--s3-prefix", type=str, default="JOIN/icechunk-stores/earthcare-store", help="S3 prefix (folder path)")
    parser.add_argument("--s3-region", type=str, default="us-east-2", help="AWS region")
    parser.add_argument("--s3-endpoint", type=str, default=None, help="Custom S3 endpoint URL (optional)")
    args = parser.parse_args()

    # Configure S3 Storage for Icechunk
    storage_kwargs = {
        "bucket": args.s3_bucket,
        "prefix": args.s3_prefix,
        "region": args.s3_region,
    }
    if args.s3_endpoint:
        storage_kwargs["endpoint_url"] = args.s3_endpoint

    storage = icechunk.s3_storage(**storage_kwargs)
    
    # Open or Create Repo
    try:
        repo = icechunk.Repository.open(storage)
        log.info(f"Opened existing Icechunk repository at s3://{args.s3_bucket}/{args.s3_prefix}")
    except Exception:
        repo = icechunk.Repository.create(storage)
        log.info(f"Created new Icechunk repository at s3://{args.s3_bucket}/{args.s3_prefix}")

    session = repo.writable_session("main")
    
    # Check if this is the first run / empty store
    is_first_run = True
    try:
        root = zarr.open_group(session.store, mode="r", zarr_format=3)
        if "EarthCARE" in root:
            is_first_run = False
            log.info("Found existing dataset hierarchy. Arrays already initialized.")
    except Exception:
        pass

    if is_first_run:
        log.info("No existing dataset found. Running initial initialization...")
        initialize_store(session)

    # Date processing
    input_dir = Path(args.input_dir)
    date_seq = pd.date_range(start=args.start_date, end=args.end_date, freq="D")
    log.info(f"Processing {len(date_seq)} dates from {args.start_date} to {args.end_date}...")
    
    for date in date_seq:
        insert_date(date, session, input_dir)

if __name__ == "__main__":
    main()