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
from zarr.codecs import Zlib
import xarray as xr
import pandas as pd

from tqdm import tqdm

import warnings
# Silence warnings about numcodecs in zarr v3
warnings.filterwarnings(
  "ignore",
  message="Numcodecs codecs are not in the Zarr version 3 specification*",
  category=UserWarning
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Workflow to virtualize GEOS-FP
# File type: netcdf
# Access: HTTPS (no authentication required)
# Notes: 
# - the data chunking changes at some point between 2019 and 2026
# - currently takes about 45 seconds to virtualize a days worth of files (24 files)

# Constants      

GEOS_BASE = (
    "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y{yyyy}/M{mm}/D{dd}/"
)

FNAME_TMPL   = "GEOS.fp.asm.tavg1_2d_slv_Nx.{date}_{time}.V01.nc4"
# For time-averaged surface level variables ('tavg1_2d_slv')

geosfp_url = "https://portal.nccs.nasa.gov/"
bucket="airborne-smce-prod-user-bucket"
region="us-west-2"
dst_prefix="JOIN/icechunk-stores/GEOS-FP-T2M"

# Helpers

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

################################################################################

# Initialize repo and virtualize data

# 1. Generate URLs

date_seq = pd.date_range(start="2019-06-01 00:30", end="2019-06-01 23:30", freq="h") 
# For time averaged data ('tavg') must start and end on half hour

log.info(f"Opening {len(geos_urls)} files virtually...")

parser = HDFParser()
src_store = obstore.store.HTTPStore(url=geosfp_url) 
registry = ObjectStoreRegistry({geosfp_url: src_store})

vds_all = open_virtual_mfdataset(geos_urls, parser=parser, registry=registry)

print(vds_all)

# If you only want T2M, you can drop other variables to keep the manifest small (to do: test how this scales to all variables)
vds_t2m = vds_all[["T2M"]]

# 2. Create IceChunk repo and virtualize data.
# If it already exists, currently need to delete and re-create.
# See potential code for appending at the end. 

dst_store = icechunk.s3_storage(
    bucket=bucket, 
    prefix=dst_prefix, 
    region=region, 
    from_env=True)

config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(geosfp_url, icechunk.http_store())
    )

log.info("Opening or creating Icechunk repository...")

repo = icechunk.Repository.open_or_create(
        storage=dst_store,
        config=config,
        authorize_virtual_chunk_access={geosfp_url: None}
    )
repo.save_config()

session = repo.writable_session("main")

#vds_all.vz.to_icechunk(session.store)
vds_t2m.vz.to_icechunk(session.store)

commit_id = session.commit(f"Virtualized GEOS-FP T2M data for {date_seq[0]} to {date_seq[-1]}")
print(f"Committed: {commit_id}")

################################################################################

# Test reading the data

log.info("Testing reading a subset of data")

session = repo.readonly_session("main")
ds = xr.open_zarr(session.store)

sub = ds.sel(lat=52.07, lon=-91.99, method="nearest").sel(
    time=slice("2019-06-01", "2019-06-01"))
print(sub["T2M"].values)

fig, ax = plt.subplots()
ds["T2M"].sel(time="2019-06-01 01:30").plot(x="lon", y="lat", ax=ax)
fig.savefig(Path("~/2m_airtemp_map.png").expanduser(), bbox_inches="tight", dpi=300)

fig, ax = plt.subplots()
ds["T2M"].sel(lat=52.07, lon=-91.99, method="nearest").plot(x="time", ax=ax)
fig.savefig(Path("~/timeseries.png").expanduser(), bbox_inches="tight", dpi=300)