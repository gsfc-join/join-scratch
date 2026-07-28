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
#from gribberish.virtualizarr import GribberishParser
import xarray as xr
import pandas as pd

import dynamical_catalog  # dynamical-catalog>=0.5.0
import dynamical_catalog._stac

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
# - virtualization is not an option directly from NOAA because of .gzip format, must unzip in local S3 first
# - update (7/10/26): there is an existing icechunk store by dynamical.org (https://dynamical.org/catalog/noaa-mrms-conus-analysis-hourly/)
# - update (7/28/26): the existing store can be accessed with icechunk functionality, without using dynamical code

# Call existing icechunk store through dynamical catalog

#ds = dynamical_catalog.open("noaa-mrms-conus-analysis-hourly", chunks=None)

# Alternatively, call using standard icechunk functionality
# url = "s3://dynamical-noaa-mrms/noaa-mrms-conus-analysis-hourly/v0.3.0.icechunk/"

storage = icechunk.s3_storage(
    bucket="dynamical-noaa-mrms",                        
    prefix="noaa-mrms-conus-analysis-hourly/v0.3.0.icechunk/",    
    anonymous=True                                
)

repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="main") # Check out the main branch or a specific commit

ds = xr.open_zarr(session.store)


# Access data for a specific location and time step
subset_range = ds["precipitation_surface"].sel(
    latitude=40, 
    longitude=-90, 
    method="nearest"
).sel(
    time=slice("2026-01-24T00", "2026-02-19T12")
).compute()

print("\n--- Data for Time Range ---")
print(subset_range.values)

# See mrms-01.5-virtualize.py for alternative to load, unzip and virtualize files.