#!/usr/bin/env python

from join_scratch.utils.s3 import ic_s3_credentials

from pathlib import Path

import icechunk as ic
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
from matplotlib import pyplot as plt

bucket = "airborne-smce-prod-user-bucket"
region = "us-west-2"
dst_prefix = "JOIN/icechunk-stores/VIIRS/VJ110A1F"

dst_store = ic.s3_storage(
    bucket=bucket,
    prefix=dst_prefix,
    region=region,
    from_env=True
)

src_prefix_url = f"s3://{bucket}/JOIN/VIIRS/VJ110A1F/"

credentials = ic.credentials.containers_credentials(
    {src_prefix_url: ic_s3_credentials()}
)

repo_ro = ic.Repository.open(
    dst_store,
    authorize_virtual_chunk_access=credentials
)
session_ro = repo_ro.readonly_session("main")

vds_test = xr.open_zarr(session_ro.store)

modis_ccrs = ccrs.Sinusoidal()

# Test region around Minneapolis
# xmin, ymin, xmax, ymax
test_bbox = [-95.6326, 43.5113, -85.4592, 48.1318]
test_bbox_modis = modis_ccrs.transform_points(
    ccrs.PlateCarree(),
    np.array(test_bbox[0:4:2]),
    np.array(test_bbox[1:4:2])
)

# NOTE: Slice from **top to bottom** in the Y dimension (`slice(ymax, ymin)`) 
# because the Y coordinate is stored in descending order.
vds_test_sub = vds_test.sel(
    XDim = slice(test_bbox_modis[0,0], test_bbox_modis[1,0]),
    YDim = slice(test_bbox_modis[1,1], test_bbox_modis[0,1])
)

fig, ax = plt.subplots()
vds_test_sub["CGF_NDSI_Snow_Cover"].sel(time="2019-01-01").plot(x="XDim", y="YDim", ax=ax)
fig.savefig(Path("~/viirs_map_jan1.png").expanduser(), bbox_inches="tight", dpi=300)
