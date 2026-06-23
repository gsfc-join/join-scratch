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

x_coord_array = np.concat(x_coord_list)

# Get all unique Y coordinates from vertical tiles
y_coord_list = []
for i in tqdm(range(0, vmax)):
    url = src_url_df.query(f"vertical_tile == {i}")["url"].iloc[0]
    tvds = open_viirs_vds(url)
    ycoords = tvds["YDim"].values
    y_coord_list.append(ycoords)

y_coord_array = np.concat(y_coord_list)

# Construct a complete global empty xarray dataset from all possible VIIRS grid 
# coordinates
# viirs_sine_ds = xr.Dataset(coords={
#     "XDim": x_coord_array,
#     "YDim": y_coord_array
# })
# shp = (viirs_sine_ds.sizes["YDim"], viirs_sine_ds.sizes["XDim"])
# empty = da.full(shp, np.nan, chunks = (1000, 1000))
reference = tvds["CGF_NDSI_Snow_Cover"]
empty_marr = reference.variable.data.with_fill_value_only(reference.attrs["_FillValue"])

# Now, construct a complete empty VIIRS grid for all tiles
empty_vds_hlist = []
for h in range(0, hmax):
    empty_vds_vlist = []
    for v in range(0, vmax):
        empty_da = xr.DataArray(
                empty_marr,
                dims=reference.dims,
                coords={
                    "XDim": x_coord_list[h],
                    "YDim": y_coord_list[v]
                },
                attrs=reference.attrs,
                name=reference.name
        )
        empty_vds_vlist.append(empty_da.copy())
    empty_vds_hlist.append(empty_vds_vlist.copy())

viirs_sine_da = xr.concat((xr.concat(vl, "YDim") for vl in empty_vds_hlist), "XDim")
viirs_sine_ds = viirs_sine_da.to_dataset(name="CGF_NDSI_Snow_Cover")

# Open one VIIRS granule 
vds1 = open_viirs_vds(src_url_df["url"].iloc[0])

viirs_sine_ds["CGF_NDSI_Snow_Cover"].loc[{
    "XDim": vds1["XDim"],
    "YDim": vds1["YDim"]
}] = vds1["CGF_NDSI_Snow_Cover"]

# Get all the X coordinates at the equator (vertical tile)
meridian_tiles = src_url_df.query("horizontal_tile == 18 and doy == 1")
meridian_vds = [open_viirs_vds(url) for url in meridian_tiles["url"]]

equator_tiles = src_url_df.query("vertical_tile == 18 and doy == 1")

# For now, just virtualize

left = open_viirs_vds(src_url_df.query("horizontal_tile==0")["url"].iloc[0])
right = open_viirs_vds(src_url_df.query("horizontal_tile==35")["url"].iloc[-1])
top = open_viirs_vds(src_url_df.query("vertical_tile==0")["url"].iloc[0])
bot = open_viirs_vds(src_url_df.query("vertical_tile==17")["url"].iloc[-1])

left["XDim"].min()
right["XDim"].max()

src_url_df.max()
bot = open_viirs_vds()
bot = open_viirs_vds(str(src_url_df
                      .query("horizontal_tile == 35")
                      .query("vertical_tile == 10")
                      .query("doy == 1")["url"].iloc[0]))

vds1["Daily_NDSI_Snow_Cover"]

src_urls[0]
src_urls[1]
vds1 = open_viirs_vds(src_url_df.loc[0, :]["url"])
vds2 = open_viirs_vds(src_url_df.loc[1, :]["url"])
vds3 = open_viirs_vds(src_url_df.loc[2, :]["url"])

vds0 = open_viirs_vds(src_url_df.loc[0, "url"])
vdslast = open_viirs_vds(src_url_df.loc[])

vds_list = [open_viirs_vds(url) for url in src_url_df.loc[0:10, "url"]]
vds_combined = xr.combine_by_coords(vds_list[0:6])

df_sub = src_url_df[src_url_df["doy"] == 1][src_url_df["horizontal_tile"] < 3]

h0 = df_sub.loc[df_sub["horizontal_tile"] == 0]["url"]
vds_h0 = xr.concat([open_viirs_vds(url) for url in h0], "YDim")
h1 = df_sub.loc[df_sub["horizontal_tile"] == 1]["url"]
vds_h1 = xr.concat([open_viirs_vds(url) for url in h1], "YDim")

vds_h01 = xr.concat([vds_h0, vds_h1], "XDim")

vds4 = open_viirs_vds(src_url_df.loc[4, :]["url"])

vds

vds123 = xr.concat([vds1, vds2, vds3], "YDim")

df1 = src_url_df.loc[0:2, :]
df1["xds"] = df1["url"].apply(open_viirs_vds)
df1

src_url_df["xds"] = src_url_df

vds12 = xr.concat([vds1, vds2])

# vds_all = open_virtual_mfdataset(src_urls, parser=parser, registry=registry)
