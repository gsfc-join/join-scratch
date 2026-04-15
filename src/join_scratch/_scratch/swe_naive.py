#!/usr/bin/env python

import tempfile
from obstore.fsspec import FsspecStore
from obstore.store import S3Store

import numpy as np

import xarray as xr
import h5py

import pyproj
import xesmf

BUCKET = "airborne-smce-prod-user-bucket"

joinbucket = S3Store(BUCKET, region="us-west-2", prefix="JOIN")

s3 = FsspecStore("s3", region="us-west-2")

lis_path = f"s3://{BUCKET}/JOIN/lis_input_NMP_1000m_missouri.nc"
lis_ds = xr.open_dataset(s3.open(lis_path), engine="h5netcdf")

lis_ds_sub = lis_ds[["lat", "lon"]]

# LIS LCC projection
# lcc_proj = pyproj.Proj(
#         proj = "lcc",
#         lat_1 = ..., lat_2 = ...,
#         lon_1 = ..., lon_2 = ...,
#         datum = "WGS84"
#         )

amsr2_search = joinbucket.list("AMSR2/")
amsr2_files = sorted([x["path"] for page in amsr2_search for x in page])

amsr2_fname = amsr2_files[0]
amsr2_url = f"s3://{BUCKET}/JOIN/{amsr2_fname}"

# amsr2_hf = h5py.File(s3.open(amsr2_url), "r")
# list(amsr2_hf.keys())
# amsr2_hf["Average Number"]

amsr2_ds = (
    xr.open_dataset(
        s3.open(amsr2_url),
        engine="h5netcdf",
        phony_dims="sort",
    )
    .rename_dims(
        {
            "phony_dim_0": "lat",
            "phony_dim_1": "lon",
            "phony_dim_2": "orbit_pass",
        }
    )
    .assign_coords(
        {
            "lat": np.linspace(89.95, -89.95, 1800),
            "lon": np.linspace(-179.95, 179.95, 3600),
            "orbit_pass": ["Ascending", "Descending"],
        }
    )
    .sortby(["lat", "lon"])
)
amsr2_ds = amsr2_ds.where(amsr2_ds["Geophysical Data"] >= 0)

lat_slice = slice(np.floor(lis_ds_sub["lat"].min()), np.ceil(lis_ds_sub["lat"].max()))
lon_slice = slice(np.floor(lis_ds_sub["lon"].min()), np.ceil(lis_ds_sub["lon"].max()))

# amsr2_ds_sub = amsr2_ds.sel(lat = lat_slice, lon = lon_slice)
amsr2_ds_sub = amsr2_ds.sel(lat = lat_slice, lon = lon_slice)
amsr2_grid = amsr2_ds_sub[["lat", "lon"]]

regridder = xesmf.Regridder(
        amsr2_grid,
        lis_ds_sub,
        method="bilinear",
        periodic=True
)
out_path = f"s3://{BUCKET}/JOIN/amsr2-lis-weights.nc"
out_nc = s3.open(out_path, "w")
regridder.to_netcdf(out_nc, engine="h5netcdf")

amsr2_regridded = regridder(amsr2_ds_sub)

# from matplotlib import pyplot as plt

# amsr2_ds["Geophysical Data"].isel(orbit_pass=0).plot(x = "longitude", y = "latitude")
# plt.savefig("~/_figures/amsr2_0.png")
# plt.close()
#
# amsr2_ds["Geophysical Data"].isel(orbit_pass=1).plot(x = "longitude", y = "latitude")
# plt.savefig("~/_figures/amsr2_1.png")
# plt.close()

amsr2_ds["Geophysical Data"]

amsr2_ds.attrs

# amsr2_search = joinbucket.list("AMSR2/")
# amsr2_files = sorted([x["path"] for page in amsr2_search for x in page])
