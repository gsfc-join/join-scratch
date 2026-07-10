#!/usr/bin/env python

import os
# Silence icechunk rust warnings. Must be set before importing icechunk.
os.environ.setdefault("RUST_LOG", "error")

import icechunk

from virtualizarr import open_virtual_dataset, open_virtual_mfdataset
from virtualizarr.parsers import HDFParser

import obstore
from obstore.store import HTTPStore
from obspec_utils.registry import ObjectStoreRegistry

import shutil
import io
from pathlib import Path
import logging

import numpy as np
import h5py
import zarr
from zarr.codecs import Zlib, Shuffle
import gribberish # recent release for compatibility of grib2 with virtualizarr
from gribberish.virtualizarr import GribberishParser
import xarray as xr
import pandas as pd

import dynamical_catalog  # dynamical-catalog>=0.5.0

from matplotlib import pyplot as plt

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import warnings
# Silence warnings about numcodecs in zarr v3
warnings.filterwarnings(
  "ignore",
  message="Numcodecs codecs are not in the Zarr version 3 specification*",
  category=UserWarning
)

# Workflow to virtualize MRMS
# File type: grib2 (but is .gz zipped)
# Access: S3 bucket (Alternative: HTTPS endpoint at "https://mrms.ncep.noaa.gov/")
# Notes:
# - see examples for working with MRMS data: https://projectpythia.org/mrms-cookbook/
# - virtualization is not an option directly from NOAA because of .gzip format, must unzip in local S3 first...
# - update (7/10/26): there is zarr formatted data stored by dynamical.org (https://dynamical.org/catalog/noaa-mrms-conus-analysis-hourly/)

#### THIS DOESN'T WORK YET ####

# Constants

MRMS_BUCKET = "noaa-mrms-pds/CONUS" # only require CONUS data for JOIN
MRMS_PRODUCT = "MultiSensor_QPE_01H_Pass2_00.00" # update to reflect info from Meloe

mrms_url = "https://portal.nccs.nasa.gov/"
USER_BUCKET = "airborne-smce-prod-user-bucket"
REGION = "us-west-2"
data_prefix = "JOIN/MRMS"
icechunk_prefix = "JOIN/icechunk-stores/MRMS"

# Helpers

def mrms_s3_path(date: pd.Timestamp) -> str:
    """Return the S3 path for MRMS QPE for a given date/hour."""
    # MRMS folders are organized by YYYYMMDD
    date_folder = date.strftime("%Y%m%d")
    
    # MRMS filenames use YYYYMMDD-HHMMSS
    # Assuming we want the top of the hour (HH:00:00)
    time_str = date.strftime("%H0000") 
    
    filename = f"{MRMS_PRODUCT}_{date_folder}-{time_str}.grib2.gz"
    
    return f"s3://{MRMS_BUCKET}/{MRMS_PRODUCT}/{date_folder}/{filename}"

def user_s3_path(date: pd.Timestamp) -> str:
    """Return the destination path in your bucket for the uncompressed file."""
    
    date_folder = date.strftime("%Y%m%d")
    time_str = date.strftime("%H0000") 
    
    filename = f"{MRMS_PRODUCT}_{date_folder}-{time_str}.grib2"
    
    return f"{USER_BUCKET}/{data_prefix}/{date_folder}/{filename}"

def decompress_to_user_bucket(date_seq: pd.DatetimeIndex, noaa_fs: s3fs.S3FileSystem, user_fs: s3fs.S3FileSystem) -> list:
    """Streams .gz from NOAA, decompresses, and writes to user bucket. Returns list of new S3 URIs."""
    uncompressed_urls = []
    
    for date in tqdm(date_seq, desc="Transferring & Decompressing"):
        src_path = mrms_s3_path(date)
        dst_path = user_s3_path(date)
        s3_uri = f"s3://{dst_path}"
        uncompressed_urls.append(s3_uri)
        
        # Skip if we already decompressed it previously
        if user_fs.exists(dst_path):
            log.info(f"File already exists in user bucket, skipping transfer: {dst_path}")
            continue
            
        try:
            # Stream from NOAA, decompress in-flight, write to User Bucket
            with noaa_fs.open(src_path, 'rb') as f_in:
                with gzip.GzipFile(fileobj=f_in, mode='rb') as gz:
                    with user_fs.open(dst_path, 'wb') as f_out:
                        shutil.copyfileobj(gz, f_out)
        except Exception as e:
            log.error(f"Failed to transfer {src_path}: {e}")
            
    return uncompressed_urls

# 1. Setup filesystems

noaa_fs = s3fs.S3FileSystem(anon=True)
user_fs = s3fs.S3FileSystem(anon=False) # Uses your environment's AWS credentials

# 2. Decompress and host files in your bucket

#date_seq = pd.date_range(start="2026-01-24 00:30", end="2026-02-19 23:30", freq="h") # check how often data is stored
date_seq = pd.date_range(start="2026-01-24 00:30", end="2026-01-24 23:30", freq="h") # check how often data is stored

uncompressed_urls = decompress_to_user_bucket(date_seq, noaa_fs, user_fs)

# 3. Create Virtualizarr Manifests mapping to the newly hosted files

log.info(f"Opening {len(uncompressed_urls)} files virtually...")

# Setup Obstore registry for the user bucket so Virtualizarr can read the headers
# Note: obstore S3 access requires AWS credentials in the environment
store = obstore.store.S3Store(USER_BUCKET, region=REGION)
registry = ObjectStoreRegistry({f"s3://{USER_BUCKET}": store})
parser = GribberishParser()
    
# Generate the virtual dataset combining all uncompressed files
vds_all = open_virtual_mfdataset(
    uncompressed_urls, 
    parser=parser, 
    registry=registry,
    concat_dim='time',
    combine='nested'
)
    
print("\nVirtual Dataset Created:")
print(vds_all)

# 4. Initialize and write to Icechunk
log.info("Opening or creating Icechunk repository...")
    
# Ensure Icechunk has permission to read the virtual chunks from your bucket
dst_store = icechunk.s3_storage(bucket=USER_BUCKET, prefix=ICECHUNK_PREFIX, region=REGION)
config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(f"s3://{USER_BUCKET}", icechunk.s3_store(region=REGION))
)
    
repo = icechunk.Repository.open_or_create(
    storage=dst_store,
    config=config,
    authorize_virtual_chunk_access={f"s3://{USER_BUCKET}": None}
)
    
session = repo.writable_session("main")
    
# Write the virtual manifest to the Icechunk store
vds_all.vz.to_icechunk(session.store)
    
commit_id = session.commit(f"Virtualized MRMS from {date_seq[0]} to {date_seq[-1]}")
log.info(f"Successfully committed to Icechunk! Commit ID: {commit_id}")
