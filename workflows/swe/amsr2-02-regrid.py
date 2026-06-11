#!/usr/bin/env python

import os

# Silence icechunk rust warnings. Must be set before importing icechunk.
os.environ.setdefault("RUST_LOG", "error")

from pathlib import Path

from tqdm import tqdm
import icechunk
import xarray as xr
import numpy as np
import scipy
import pandas as pd

import subprocess

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# AMSR2 fill values
FILL_MISSING = np.int16(-32768)
FILL_NOOBS = np.int16(-32767)
SCALE_FACTOR = 0.1

def create_scrip_grid(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    mask2d: np.ndarray | None = None,
    title: str = "grid",
) -> xr.Dataset:
    ny, nx = lat2d.shape
    ncells = ny * nx

    lat_flat = lat2d.flatten()
    lon_flat = lon2d.flatten()

    # Compute corner coordinates (simple ±0.5 cell approximation for regular grids)
    dlat = np.gradient(lat2d, axis=0)
    dlon = np.gradient(lon2d, axis=1)

    lat_corners = np.stack(
        [
            lat_flat - dlat.flatten() / 2,
            lat_flat - dlat.flatten() / 2,
            lat_flat + dlat.flatten() / 2,
            lat_flat + dlat.flatten() / 2,
        ],
        axis=1,  # Stack along columns to get shape (grid_size, grid_corners)
    )
    lon_corners = np.stack(
        [
            lon_flat - dlon.flatten() / 2,
            lon_flat + dlon.flatten() / 2,
            lon_flat + dlon.flatten() / 2,
            lon_flat - dlon.flatten() / 2,
        ],
        axis=1,  # Stack along columns to get shape (grid_size, grid_corners)
    )

    # Handle the mask logic
    if mask2d is not None:
        imask = mask2d.flatten().astype(np.int32)
    else:
        imask = np.ones(ncells, dtype=np.int32)

    # Build the Xarray Dataset matching SCRIP specs
    ds = xr.Dataset(
        data_vars={
            "grid_dims": (["grid_rank"], np.array([nx, ny], dtype=np.int32)),
            "grid_center_lat": (
                ["grid_size"],
                lat_flat,
                {"units": "degrees"},
            ),
            "grid_center_lon": (
                ["grid_size"],
                lon_flat,
                {"units": "degrees"},
            ),
            "grid_corner_lat": (
                ["grid_size", "grid_corners"],
                lat_corners,
                {"units": "degrees"},
            ),
            "grid_corner_lon": (
                ["grid_size", "grid_corners"],
                lon_corners,
                {"units": "degrees"},
            ),
            "grid_imask": (["grid_size"], imask, {"units": "unitless"}),
        },
        attrs={
            "title": title,
        },
    )

    return ds


# Load weights
def load_weights(fname: str | Path):
    ds = xr.open_dataset(fname, engine="h5netcdf")
    S = ds["S"].values
    row = ds["row"].values - 1
    col = ds["col"].values - 1
    n_dst = int(ds["dst_grid_dims"].prod().values)
    n_src = int(ds["src_grid_dims"].prod().values)
    W = scipy.sparse.csr_matrix((S, (row, col)), shape=(n_dst, n_src))
    return W


def regrid_slice(
    data: np.ndarray,
    W: scipy.sparse.csr_matrix,
    dst_shape: tuple[int, ...]
) -> np.ndarray:
    """Regrid a single 2-D source field using precomputed ESMF weights.

    Fill-valued pixels are masked to NaN before the sparse matrix multiply so
    they cannot infect neighbouring destination cells.  Destination cells with
    no valid source contribution remain NaN.

    Parameters
    ----------
    data:
        2-D array of raw int16 source values, shape ``(NLAT, NLON)``.
        Assumed layout: row 0 = lat 89.95°N, last row = lat −89.95°S (north-first),
        columns: lon 0.05°E … 359.95°E.
    W:
        CSR sparse weight matrix, shape ``(n_dst, n_src)``.
    dst_shape:
        ``(ny_lis, nx_lis)`` — shape of the destination grid.
    fill_values:
        Raw int16 values that indicate missing/no-observation; these are
        masked to NaN before regridding.

    Returns
    -------
    Float32 array of shape *dst_shape* with regridded physical values
    (i.e. raw × SCALE_FACTOR).  Cells with no valid source data are NaN.
    """
    float_data = data

    src_flat = float_data.flatten()
    valid = np.isfinite(src_flat)

    # To avoid NaN contamination in the sparse multiply, replace NaN with 0
    # and use a separate weight-sum array to detect cells with zero valid weight.
    src_clean = np.where(valid, src_flat, 0.0)
    valid_weight = np.where(valid, 1.0, 0.0)

    dst_vals = W @ src_clean
    dst_wsum = W @ valid_weight

    # Normalise by the actual sum of weights that came from valid pixels.
    with np.errstate(invalid="ignore", divide="ignore"):
        dst_norm = np.where(dst_wsum > 0, dst_vals / dst_wsum, np.nan)

    return dst_norm.reshape(dst_shape)

################################################################################
# Read the LIS grid

bucket_name = "airborne-smce-prod-user-bucket"
lis_prefix = "JOIN/lis_input_NMP_1000m_missouri.nc"
lis_url = f"s3://{bucket_name}/{lis_prefix}"
lis = xr.open_dataset(lis_url, engine="h5netcdf")

tmp_dir = Path("/tmp")

lis_lats = lis["lat"].values
lis_lons = lis["lon"].values

assert lis_lats.shape == lis_lons.shape

lis_scrip = create_scrip_grid(lis_lats, lis_lons, title = "LIS NMP 1000m Missouri")
lis_scrip_nc4 = tmp_dir / "lis-scrip.nc4"
lis_scrip.to_netcdf(lis_scrip_nc4, format="NETCDF4")

################################################################################
# Open AMSR2 store
jaxa_gportal_url = "https://gportal.jaxa.jp/"
repo = icechunk.Repository.open(
    icechunk.s3_storage(
        bucket="airborne-smce-prod-user-bucket",
        prefix="JOIN/icechunk-stores/GCOM-W1-AMSR2-L3-SND",
    ),
    authorize_virtual_chunk_access={jaxa_gportal_url: None}
)
session = repo.readonly_session("main")

amsr2_ds = xr.open_zarr(session.store)

amsr2_lats = amsr2_ds["lat"].values
amsr2_lons = amsr2_ds["lon"].values
amsr2_lons_2d, amsr2_lats_2d = np.meshgrid(amsr2_lons, amsr2_lats)   
amsr2_scrip = create_scrip_grid(amsr2_lats_2d, amsr2_lons_2d, title = "GCOM-W1-AMSR2-L3-SND")
amsr2_scrip_nc4 = tmp_dir / "amsr2-script.nc4"
amsr2_scrip.to_netcdf(amsr2_scrip_nc4, format="NETCDF4")

################################################################################
# Generate ESMF weights

esmf_out = tmp_dir / "weights.nc4"
if not esmf_out.exists():
    cmd = [
        "ESMF_RegridWeightGen",
        "--source",
        str(amsr2_scrip_nc4),
        "--destination",
        str(lis_scrip_nc4),
        "--weight",
        str(esmf_out),
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
    print(result.stdout)
else:
    log.info("Using existing weights at %s", str(esmf_out))

W = load_weights(esmf_out)

################################################################################
# Create result dataset
time_values = pd.date_range(start="2019-01-01", end="2019-01-08", freq="D")

reference = lis["TBOT"]
dst_shape = reference.shape
regridded_slices = []

# NOTE: This can be naively parallelized
for t in tqdm(time_values):
    dat = (amsr2_ds["geophysical_data"].sel(time=t, method="nearest")
           .sel(orbit="Descending", band="snow_depth")).values
    regridded = regrid_slice(dat, W, dst_shape)
    da_slice = xr.full_like(reference, fill_value=np.nan)
    da_slice.values = regridded
    regridded_slices.append(da_slice)

result_da = xr.concat(regridded_slices, dim=pd.Index(time_values, name="time"))

result = result_da.to_dataset(name = "snow_depth_amsr2")
s3_tmp = f"s3://{bucket_name}/JOIN/outputs-preliminary"
result.to_zarr(f"{s3_tmp}/amsr2-regridded.zarr")
