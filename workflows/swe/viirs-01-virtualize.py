#!/usr/bin/env python

from pathlib import Path
import os
import re

# Silence icechunk rust warnings. Must be set before importing icechunk.
os.environ.setdefault("RUST_LOG", "error")

from join_scratch.utils.s3 import s3_store_config, list_s3

import obstore
from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr import open_virtual_dataset, open_virtual_mfdataset
from virtualizarr.parsers import HDFParser
from virtualizarr.manifests import ManifestArray

import icechunk as ic

from tqdm import tqdm
import xarray as xr
import pandas as pd
import numpy as np
import dask.array as da

from matplotlib import pyplot as plt

bucket = "airborne-smce-prod-user-bucket"
region = "us-west-2"
src_bucket_url = f"s3://{bucket}"
dst_bucket_url = f"s3://{bucket}"

src_prefix_path = "JOIN/VIIRS/VJ110A1F"

src_store = obstore.store.S3Store(bucket=bucket, config=s3_store_config())
registry = ObjectStoreRegistry({src_bucket_url: src_store})

# Get list of files
src_files = sorted(f for f in list_s3(src_store, src_prefix_path) if f.endswith(".h5"))
src_urls = [f"{src_bucket_url}/{f}" for f in src_files]

# Parse these into a data frame for easier processing
src_url_pat = re.compile(r'A(\d{4})(\d{3})\.h(\d{2})v(\d{2})')
parsed_urls = []
for url in src_urls:
    match = src_url_pat.search(url)
    if not match:
        continue
    year, doy, htile, vtile = match.groups()
    date_obj = pd.to_datetime(f"{year}{doy}", format="%Y%j")
    parsed_urls.append({
        "date": date_obj,
        "year": int(year),
        "doy": int(doy),
        "horizontal_tile": int(htile),
        "vertical_tile": int(vtile),
        "url": url
    })
src_url_df = pd.DataFrame(parsed_urls)

# Build virtual zarr store of all the VIIRS data

# Try opening one VIIRS file
def open_viirs_vds(src_url):
    # src_url = src_urls[0]
    VIIRS_HDF_ROOT = "HDFEOS/GRIDS/VIIRS_Grid_IMG_2D"
    dim_rename = {
            f"{VIIRS_HDF_ROOT}/YDim": "YDim",
            f"{VIIRS_HDF_ROOT}/XDim": "XDim"
    }
    vds_data = open_virtual_dataset(src_url, registry, HDFParser(
        group=f"{VIIRS_HDF_ROOT}/Data Fields"
    )).rename(dim_rename)
    vds_coords = open_virtual_dataset(src_url, registry, HDFParser(group=VIIRS_HDF_ROOT))
    vds = xr.merge([vds_data, vds_coords])
    return vds.copy()

# Create an Xarray dataset of the entire
hmax = 35
vmax = 17

# Get all unique X coordinates from horizontal tiles
x_coord_list = []
for i in tqdm(range(0, hmax)):
    url = src_url_df.query(f"horizontal_tile == {i}")["url"].iloc[0]
    tvds = open_viirs_vds(url)
    xcoords = tvds["XDim"].values
    x_coord_list.append(xcoords)

# Get all unique Y coordinates from vertical tiles
y_coord_list = []
for i in tqdm(range(0, vmax)):
    url = src_url_df.query(f"vertical_tile == {i}")["url"].iloc[0]
    tvds = open_viirs_vds(url)
    ycoords = tvds["YDim"].values
    y_coord_list.append(ycoords)

def virtualize_viirs_date(date_dat: pd.DataFrame, empty_marr: ManifestArray):
    vds_hlist = []
    for h in tqdm(range(0, hmax), "horizontal"):
        vds_vlist = []
        for v in tqdm(range(0, vmax), desc="vertical", leave=False):
            # Look for a VIIRS granule.
            hvdat = date_dat.query("horizontal_tile == @h and vertical_tile == @v")
            if hvdat.empty:
                # print(f"h{h} v{v} - empty")
                item = xr.DataArray(
                    empty_marr,
                    dims=reference.dims,
                    coords={
                        "XDim": x_coord_list[h],
                        "YDim": y_coord_list[v]
                    },
                    attrs=reference.attrs,
                    name=reference.name
                ).to_dataset()
            else:
                # print(f"h{h} v{v} - got item")
                item = open_viirs_vds(hvdat["url"].iloc[0])[["CGF_NDSI_Snow_Cover"]]
            vds_vlist.append(item.copy())
        vds_hlist.append(vds_vlist.copy())
    viirs_complete = xr.concat((xr.concat(vl, "YDim") for vl in vds_hlist), "XDim")
    return viirs_complete.copy()

# Variable that we are targeting
reference = tvds["CGF_NDSI_Snow_Cover"]

# Empty ManifestArray that we will use as a template for situations where VIIRS 
# data are not available.
empty_marr = reference.variable.data.with_fill_value_only(reference.attrs["_FillValue"])

# Now, construct a complete empty VIIRS grid for all tiles
src_url_df = src_url_df.set_index("date")

dates = pd.date_range("2019-01-01", "2019-01-07", freq="D")
dates = dates[0:2]

# vds1 = virtualize_viirs_date(src_url_df.loc['2019-01-01'], empty_marr)
# vds2 = virtualize_viirs_date(src_url_df.loc[pd.to_datetime("2019-01-02")], empty_marr)

vds_complete = xr.concat((
    virtualize_viirs_date(src_url_df.loc[d], empty_marr) for d in tqdm(dates, desc="Dates")
), dim=pd.Index(dates, name="time"))

# Create icechunk store
vds12.vz.to_kerchunk(Path("~/viirs.json").expanduser(), format = "json")

# Try reading 
vds_test = xr.open_dataset("~/viirs.json", engine="kerchunk")

import cartopy.crs as ccrs
modis_

fig, ax = plt.subplots()
vds_test["CGF_NDSI_Snow_Cover"].sel(time="2019-01-01").plot(x="XDim", y="YDim", ax=ax)
fig.savefig("~/viirs_map_jan1.png", bbox_inches="tight", dpi=300)

fig, ax = plt.subplots()
vds_test["CGF_NDSI_Snow_Cover"].sel(time="2019-01-02").plot(x="XDim", y="YDim", ax=ax)
fig.savefig("~/viirs_map_jan2.png", bbox_inches="tight", dpi=300)
