#!/usr/bin/env python

import numpy as np
import xarray as xr
from pathlib import Path
import scipy


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
def load_weights(fname: str | Path, **kwargs):
    ds = xr.open_dataset(fname, **kwargs)
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
