#!/usr/bin/env python

from pathlib import Path
import subprocess

import xarray as xr
import numpy as np
import icechunk
import pandas as pd

from tqdm import tqdm

from join_scratch.regrid.utils import (
    create_scrip_grid,
    load_weights,
    regrid_slice
)

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

tmp_dir = Path("/tmp")

bucket = "airborne-smce-prod-user-bucket"
ceda_prefix = "JOIN/icechunk-stores/CEDA"

esmf_out_s3 = f"s3://{bucket}/JOIN/cached-weights/CEDA/lis-1km-missouri.zarr"
esmf_out_tmp = tmp_dir / "ceda-lis-weights.nc4"

s3_prelim_output = f"s3://{bucket}/JOIN/outputs-preliminary/ceda-swe-regridded.zarr"

time_values = pd.date_range(start="2019-01-01", end="2019-01-08", freq="D")

################################################################################
# Read the LIS grid

lis_prefix = "JOIN/lis_input_NMP_1000m_missouri.nc"
lis_url = f"s3://{bucket}/{lis_prefix}"
lis = xr.open_dataset(lis_url, engine="h5netcdf")

lis_lats = lis["lat"].values
lis_lons = lis["lon"].values

assert lis_lats.shape == lis_lons.shape

lis_scrip = create_scrip_grid(lis_lats, lis_lons, title = "LIS NMP 1000m Missouri")
lis_scrip_nc4 = tmp_dir / "lis-scrip.nc4"
lis_scrip.to_netcdf(lis_scrip_nc4, format="NETCDF4")

################################################################################
# Open CEDA store
repo = icechunk.Repository.open(
    icechunk.s3_storage(
        bucket=bucket,
        prefix=ceda_prefix,
    ),
    authorize_virtual_chunk_access={f"s3://{bucket}/JOIN/CEDA/": None}
)
session = repo.readonly_session("main")
ceda_ds = xr.open_zarr(session.store)

ceda_lats = ceda_ds["lat"].values
ceda_lons = ceda_ds["lon"].values
ceda_lons_2d, ceda_lats_2d = np.meshgrid(ceda_lons, ceda_lats)   
ceda_scrip = create_scrip_grid(ceda_lats_2d, ceda_lons_2d, title = "GCOM-W1-ceda-L3-SND")
ceda_scrip_nc4 = tmp_dir / "ceda-scrip.nc4"
ceda_scrip.to_netcdf(ceda_scrip_nc4, format="NETCDF4")

################################################################################
# Generate ESMF weights

try:
    log.info("Trying to load existing weights at %s", str(esmf_out_s3))
    W = load_weights(esmf_out_s3)
    log.info("Loaded existing weights!")
except ValueError as e:
    log.warning(
        f"Could not read existing weights with error {str(e)}. ",
        "Recreating weights locally."
    )
    cmd = [
        "ESMF_RegridWeightGen",
        "--source",
        str(ceda_scrip_nc4),
        "--destination",
        str(lis_scrip_nc4),
        "--weight",
        str(esmf_out_tmp),
        "--method",
        "bilinear",
        "--src_type",
        "SCRIP",
        "--dst_type",
        "SCRIP",
        "--ignore_unmapped",
        "--netcdf4"
    ]
    log.info("Running %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Cache the weights to zarr on S3
    log.info("Saving weights to S3 %s", " ".join(cmd))
    xr.open_dataset(esmf_out_tmp, engine="h5netcdf").to_zarr(esmf_out_s3)
    W = load_weights(esmf_out_tmp)

################################################################################
# Create result dataset

result = (lis[["lat", "lon", "lat_b", "lon_b"]]
          .copy()
          .set_coords(["lon", "lat", "lon_b", "lat_b"]))

reference = result["lat"]
dst_shape = reference.shape
regridded_slices = []

# NOTE: This can be naively parallelized
# NOTE: May need to loop over multiple variables as well (which can also be parallelized)
log.info("Regridding time steps")
for t in tqdm(time_values):
    dat = (ceda_ds["swe"].sel(time=t, method="nearest"))
    regridded = regrid_slice(dat.values, W, dst_shape)
    reference.coords
    da_slice = xr.DataArray(
        data=regridded,
        coords=reference.coords,
        dims=reference.dims,
        attrs=dat.attrs
    )
    regridded_slices.append(da_slice)

result["swe_ceda"] = xr.concat(regridded_slices, dim=pd.Index(time_values, name="time"))

log.info("Writing result to %s", s3_prelim_output)
result.to_zarr(s3_prelim_output, mode="w")
